"""Erdős Discovery Environment for NeMo RL.

Implements EnvironmentInterface for TTT-Discover with the Erdős Minimum
Overlap Problem. Calls the NeMo Gym resource server for code execution
and reward computation.

Reference: "Learning to Discover at Test Time" (arXiv:2601.16175)

The environment:
  1. Receives LLM-generated code from the GRPO rollout
  2. Sends it to the Erdős Gym resource server for sandboxed execution + scoring
  3. Returns reward = 1/bound (or 0 on failure)
  4. Tracks best constructions and buffer statistics via metrics
"""

import logging
import math
from typing import Any, Optional

import aiohttp
import ray
import torch

from nemo_rl.environments.interfaces import EnvironmentInterface, EnvironmentReturn

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Inline reward computation (no Gym server needed for debug/testing)
# ═══════════════════════════════════════════════════════════════════

def _inline_compute_reward(response_text: str, timeout: int = 60) -> dict:
    """Compute reward directly in-process. No HTTP call needed."""
    import re
    import signal
    import builtins
    import math as _math
    import itertools as _itertools
    import functools as _functools
    import collections as _collections

    import numpy as _np
    from numpy.fft import rfft, irfft

    _ALLOWED_MODULES = frozenset({
        "numpy", "np", "math", "cmath", "random",
        "itertools", "functools", "collections", "fractions", "decimal",
    })
    _SAFE_BUILTIN_NAMES = [
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate",
        "filter", "float", "format", "int", "isinstance", "issubclass",
        "iter", "len", "list", "map", "max", "min", "next", "object",
        "print", "range", "repr", "reversed", "round", "set", "slice",
        "sorted", "str", "sum", "tuple", "type", "zip",
        "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
        "StopIteration", "RuntimeError", "NotImplementedError",
        "OverflowError", "ZeroDivisionError", "AttributeError",
    ]

    # Extract code
    code_re = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
    blocks = code_re.findall(response_text)
    code = blocks[-1].strip() if blocks else response_text.strip()

    # Build sandbox
    import random as _random
    safe_builtins = {k: getattr(builtins, k) for k in _SAFE_BUILTIN_NAMES if hasattr(builtins, k)}
    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.split(".")[0] not in _ALLOWED_MODULES:
            raise ImportError(f"Module '{name}' not allowed")
        return builtins.__import__(name, globals, locals, fromlist, level)
    safe_builtins["__import__"] = _safe_import
    namespace = {
        "__builtins__": safe_builtins,
        "np": _np, "numpy": _np, "math": _math, "random": _random,
        "itertools": _itertools, "functools": _functools, "collections": _collections,
    }

    try:
        class _Timeout(Exception):
            pass
        def _handler(s, f):
            raise _Timeout()
        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout)
        try:
            exec(compile(code, "<llm>", "exec"), namespace)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

        if "f" not in namespace:
            return {"reward": 0.0, "bound": None, "error_msg": "no variable f"}

        f = _np.asarray(namespace["f"], dtype=float).flatten()

        # Validate
        if len(f) < 1 or len(f) > 1000:
            return {"reward": 0.0, "bound": None, "error_msg": "bad length"}
        if _np.any(~_np.isfinite(f)) or _np.any(f < 0) or _np.any(f > 1):
            return {"reward": 0.0, "bound": None, "error_msg": "bad values"}
        if abs(float(_np.mean(f)) - 0.5) > 1e-3:
            return {"reward": 0.0, "bound": None, "error_msg": "bad mean"}

        # Compute bound
        n = len(f)
        F = rfft(f, n=2*n)
        autocorr = irfft(F * _np.conj(F), n=2*n)
        bound = float(2 * n * _np.max(autocorr.real) / (_np.sum(f)**2))

        if bound <= 0 or not _math.isfinite(bound):
            return {"reward": 0.0, "bound": None, "error_msg": "bad bound"}
        return {"reward": 1.0 / bound, "bound": bound, "error_msg": ""}

    except Exception as e:
        return {"reward": 0.0, "bound": None, "error_msg": str(e)[:200]}

# Type alias matching NeMo RL's convention
LLMMessageLogType = list[dict[str, Any]]
ErdosMetadata = dict[str, Any]


@ray.remote(max_restarts=-1, max_task_retries=-1)
class ErdosDiscoveryEnvironment(EnvironmentInterface[ErdosMetadata]):
    """Erdős Minimum Overlap Problem environment for GRPO training.

    Communicates with the NeMo Gym Erdős resource server via HTTP for:
      - /verify: code execution + reward computation
      - /select_state: PUCT state selection for prompts
      - /seed_session: buffer initialization
      - /compute_entropic_advantages: LOO entropic advantages
      - /update_buffer: add new discoveries to PUCT tree

    Config (under env.erdos_discovery):
        resource_server_url: Base URL of the Erdős Gym resource server.
        seed: Random seed for PUCT buffer initialization.
        num_initial_states: States to seed the buffer with.
        sandbox_timeout: Code execution timeout in seconds.
    """

    def __init__(self, config: dict):
        self.config = config
        self.resource_server_url = config.get(
            "resource_server_url", "http://localhost:8080"
        )
        self.seed = config.get("seed", None)
        self.num_initial_states = config.get("num_initial_states", 16)
        self.sandbox_timeout = config.get("sandbox_timeout", 600)
        self.request_timeout = config.get("request_timeout", 660)

        self.best_reward = 0.0
        self.best_bound = float("inf")
        self.total_verified = 0
        self.total_valid = 0
        self._session_initialized = False
        self._inline_mode = (self.resource_server_url == "inline")
        if self._inline_mode:
            logger.info("ErdosDiscovery: running in INLINE mode (no Gym server)")
            self._session_initialized = True  # No server to init

    async def _ensure_session(self):
        """Initialize the PUCT buffer on the resource server if not done."""
        if self._session_initialized:
            return
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.resource_server_url}/seed_session",
                    json={
                        "num_initial_states": self.num_initial_states,
                        "seed": self.seed,
                    },
                ) as resp:
                    data = await resp.json()
                    self.best_reward = data.get("best_initial_reward", 0.0)
                    self.best_bound = data.get(
                        "best_initial_bound", float("inf")
                    )
                    logger.info(
                        "ErdosDiscovery: seeded buffer with %d states, "
                        "best_reward=%.4f, best_bound=%.6f",
                        data.get("num_states", 0),
                        self.best_reward,
                        self.best_bound,
                    )
            self._session_initialized = True
        except Exception as e:
            logger.error("ErdosDiscovery: seed_session failed: %s", e)

    async def _verify_single(
        self,
        session: Optional[aiohttp.ClientSession],
        response_text: str,
        parent_state: Optional[list[float]] = None,
    ) -> dict:
        """Call /verify on the resource server, or compute inline."""
        if self._inline_mode:
            return _inline_compute_reward(
                response_text, timeout=self.sandbox_timeout
            )
        # Build a minimal NeMoGymResponse-like payload
        # The resource server extracts output_text from response.output_text
        body = {
            "responses_create_params": {
                "input": [{"role": "user", "content": ""}],
            },
            "response": {
                "id": "verify",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": response_text}],
                    }
                ],
                "output_text": response_text,
            },
            "parent_state": parent_state,
        }
        try:
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            async with session.post(
                f"{self.resource_server_url}/verify",
                json=body,
                timeout=timeout,
            ) as resp:
                return await resp.json()
        except Exception as e:
            logger.warning("ErdosDiscovery: verify failed: %s", e)
            return {"reward": 0.0, "bound": None, "error_msg": str(e)}

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[ErdosMetadata],
    ) -> EnvironmentReturn[ErdosMetadata]:
        """Evaluate a batch of LLM responses.

        Extracts the assistant's last message from each conversation,
        sends it to the resource server for code execution + scoring,
        returns rewards.
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Ray actors run inside an event loop — use nest_asyncio or run sync
            return self._sync_step(message_log_batch, metadata)
        else:
            return asyncio.run(
                self._async_step(message_log_batch, metadata)
            )

    def _sync_step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[ErdosMetadata],
    ) -> EnvironmentReturn[ErdosMetadata]:
        """Synchronous step for use inside running event loops (Ray actors)."""
        batch_size = len(message_log_batch)
        rewards = torch.zeros(batch_size)
        terminateds = torch.ones(batch_size)
        observations = [{}] * batch_size
        answers = [None] * batch_size
        updated_metadata = list(metadata)

        for i, message_log in enumerate(message_log_batch):
            response_text = ""
            for msg in reversed(message_log):
                if msg.get("role") == "assistant":
                    response_text = msg.get("content", "")
                    break

            if self._inline_mode:
                result = _inline_compute_reward(
                    response_text, timeout=self.sandbox_timeout
                )
            else:
                result = {"reward": 0.0, "bound": None, "error_msg": "sync mode requires inline"}

            reward = result.get("reward", 0.0)
            rewards[i] = reward
            self.total_verified += 1

            if reward > 0:
                self.total_valid += 1
                bound = result.get("bound")
                if reward > self.best_reward:
                    self.best_reward = reward
                    self.best_bound = bound or (1.0 / reward if reward > 0 else float("inf"))
                answers[i] = f"bound={bound:.6f}" if bound else f"reward={reward:.4f}"

            if i < len(updated_metadata):
                updated_metadata[i] = {
                    **updated_metadata[i],
                    "reward": reward,
                    "bound": result.get("bound"),
                    "error_msg": result.get("error_msg", ""),
                }

        return EnvironmentReturn(
            observations=observations,
            metadata=updated_metadata,
            next_stop_strings=[None] * batch_size,
            rewards=rewards,
            terminateds=terminateds,
            answers=answers,
        )

    async def _async_step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[ErdosMetadata],
    ) -> EnvironmentReturn[ErdosMetadata]:
        await self._ensure_session()

        batch_size = len(message_log_batch)
        rewards = torch.zeros(batch_size)
        terminateds = torch.ones(batch_size)  # Always single-turn
        observations = [{}] * batch_size
        answers = [None] * batch_size
        updated_metadata = list(metadata)

        if self._inline_mode:
            session = None
        else:
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.request_timeout)
            )

        try:
            tasks = []
            for i, message_log in enumerate(message_log_batch):
                # Extract the last assistant message
                response_text = ""
                for msg in reversed(message_log):
                    if msg.get("role") == "assistant":
                        response_text = msg.get("content", "")
                        break

                # Get parent_state from metadata if available
                parent_state = None
                if metadata and i < len(metadata):
                    parent_state = metadata[i].get("parent_state", None)

                tasks.append(
                    self._verify_single(session, response_text, parent_state)
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            if session is not None:
                await session.close()

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(
                    "ErdosDiscovery: verify exception for sample %d: %s",
                    i,
                    result,
                )
                continue

            reward = result.get("reward", 0.0)
            rewards[i] = reward
            self.total_verified += 1

            if reward > 0:
                self.total_valid += 1
                bound = result.get("bound", None)
                if reward > self.best_reward:
                    self.best_reward = reward
                    self.best_bound = bound or (
                        1.0 / reward if reward > 0 else float("inf")
                    )

                answers[i] = (
                    f"bound={bound:.6f}" if bound else f"reward={reward:.4f}"
                )

            # Update metadata with verification results
            if i < len(updated_metadata):
                updated_metadata[i] = {
                    **updated_metadata[i],
                    "reward": reward,
                    "bound": result.get("bound"),
                    "error_msg": result.get("error_msg", ""),
                    "best_reward_ever": result.get(
                        "best_reward_ever", self.best_reward
                    ),
                }

        return EnvironmentReturn(
            observations=observations,
            metadata=updated_metadata,
            next_stop_strings=[None] * batch_size,
            rewards=rewards,
            terminateds=terminateds,
            answers=answers,
        )

    def global_post_process_and_metrics(
        self, batch: dict
    ) -> tuple[dict, dict]:
        """Compute and return environment-level metrics."""
        valid_rate = (
            self.total_valid / max(self.total_verified, 1)
        )
        metrics = {
            "env/best_reward": self.best_reward,
            "env/best_bound": self.best_bound
            if self.best_bound < float("inf")
            else 0.0,
            "env/total_verified": self.total_verified,
            "env/valid_rate": valid_rate,
        }
        return batch, metrics

    def shutdown(self):
        """Cleanup."""
        pass
