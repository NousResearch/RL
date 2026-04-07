"""Erdős Minimum Overlap Discovery Environment — matches reference TTT-Discover implementation.

Reference: https://github.com/test-time-training/discover/blob/main/examples/erdos_min_overlap/env.py
Paper: "Learning to Discover at Test Time" (arXiv:2601.16175)

Key differences from our v1:
- Uses C5 = max(np.correlate(h, 1-h, mode="full") * dx) formulation (h over [0,2])
- Code must define run(seed=42, budget_s=1000) returning (h_values, c5_bound, n_points)
- Allows scipy, cvxpy in addition to numpy/math
- State context shows parent code + improvement direction
- reward = 1 / (1e-8 + c5_bound)
"""

import asyncio
import logging
import math
import os
import re
import signal
import time
from typing import Any, Optional

import numpy as np
import ray
import torch

from nemo_rl.environments.interfaces import (
    EnvironmentInterface,
    EnvironmentReturn,
)
from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.environments.erdos_ref_puct_sampler import (
    ErdosRefPUCTSampler,
    ErdosRefState,
    erdos_ref_state_to_prompt_state,
)

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Reference C5 verification (from TTT-Discover env.py)
# ═══════════════════════════════════════════════════════════════════

def verify_c5_solution(h_values, c5_achieved, n_points):
    """Verify a C5 solution — exact copy from reference implementation."""
    if not isinstance(h_values, np.ndarray):
        try:
            h_values = np.array(h_values, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert h_values to numpy array: {e}")

    if len(h_values.shape) != 1:
        raise ValueError(f"h_values must be 1D array, got shape {h_values.shape}")

    if h_values.shape[0] != n_points:
        raise ValueError(f"Expected h shape ({n_points},), got {h_values.shape}")

    if not np.all(np.isfinite(h_values)):
        raise ValueError("h_values contain NaN or inf values")

    if np.any(h_values < 0) or np.any(h_values > 1):
        raise ValueError(f"h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")

    n = n_points
    target_sum = n / 2.0
    current_sum = np.sum(h_values)

    if current_sum != target_sum:
        h_values = h_values * (target_sum / current_sum)
        if np.any(h_values < 0) or np.any(h_values > 1):
            raise ValueError(
                f"After normalization, h(x) is not in [0, 1]. "
                f"Range: [{h_values.min()}, {h_values.max()}]"
            )

    dx = 2.0 / n_points
    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    computed_c5 = np.max(correlation)

    if not np.isfinite(computed_c5):
        raise ValueError(f"Computed C5 is not finite: {computed_c5}")

    if not np.isclose(computed_c5, c5_achieved, atol=1e-4):
        raise ValueError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")

    return computed_c5


# ═══════════════════════════════════════════════════════════════════
# Sandbox execution
# ═══════════════════════════════════════════════════════════════════

_ALLOWED_MODULES = frozenset({
    "numpy", "np", "math", "cmath", "random", "scipy", "cvxpy",
    "itertools", "functools", "collections", "fractions", "decimal",
    "copy", "operator", "time",
})


def _execute_run_function(code: str, timeout: int = 1000, n_cpus: int = 2) -> dict:
    """Execute code that defines run(), call it, verify the result.

    Matches the reference SandboxRewardEvaluator flow.
    """
    import builtins

    _SAFE_BUILTIN_NAMES = [
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate",
        "filter", "float", "format", "int", "isinstance", "issubclass",
        "iter", "len", "list", "map", "max", "min", "next", "object",
        "print", "range", "repr", "reversed", "round", "set", "slice",
        "sorted", "str", "sum", "tuple", "type", "zip", "True", "False",
        "None", "complex", "frozenset", "bytes", "bytearray", "memoryview",
        "property", "staticmethod", "classmethod", "super", "hash", "id",
        "input", "ord", "chr", "hex", "oct", "bin", "pow",
        "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
        "StopIteration", "RuntimeError", "NotImplementedError",
        "OverflowError", "ZeroDivisionError", "AttributeError",
        "ImportError", "FileNotFoundError", "OSError", "ArithmeticError",
    ]

    safe_builtins = {k: getattr(builtins, k) for k in _SAFE_BUILTIN_NAMES
                     if hasattr(builtins, k)}

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        base = name.split(".")[0]
        if base not in _ALLOWED_MODULES:
            raise ImportError(f"Module '{name}' not allowed")
        return builtins.__import__(name, globals, locals, fromlist, level)

    safe_builtins["__import__"] = _safe_import

    import random as _random
    namespace = {
        "__builtins__": safe_builtins,
        "np": np,
        "numpy": np,
        "math": math,
        "random": _random,
    }

    # Add evaluate_erdos_solution to namespace (reference injects this)
    def _evaluate_erdos_solution(h_values, c5_bound, n_points):
        verify_c5_solution(h_values, c5_bound, n_points)
        return float(c5_bound)

    namespace["evaluate_erdos_solution"] = _evaluate_erdos_solution

    stdout_capture = []
    original_print = builtins.print
    def capturing_print(*args, **kwargs):
        import io
        buf = io.StringIO()
        kwargs["file"] = buf
        original_print(*args, **kwargs)
        stdout_capture.append(buf.getvalue())
    namespace["__builtins__"]["print"] = capturing_print

    try:
        # Use multiprocessing for timeout (signal.alarm doesn't work in Ray actor threads)
        import multiprocessing as mp
        import pickle

        def _run_in_subprocess(code, ns_pickle, result_queue):
            """Execute code in a subprocess with proper timeout support."""
            import signal as _signal
            namespace = pickle.loads(ns_pickle)

            class _Timeout(Exception):
                pass
            def _handler(s, f):
                raise _Timeout("timeout")
            _signal.signal(_signal.SIGALRM, _handler)
            _signal.alarm(timeout)
            try:
                exec(compile(code, "<llm>", "exec"), namespace)
                if "run" not in namespace:
                    result_queue.put({"error": "No 'run' function defined"})
                    return
                out = namespace["run"](seed=42, budget_s=timeout)
                result_queue.put({"result": out})
            except _Timeout:
                result_queue.put({"error": f"Execution timed out after {timeout}s"})
            except Exception as e:
                result_queue.put({"error": f"{type(e).__name__}: {str(e)[:300]}"})

        # Run in subprocess with signal.alarm for clean timeout.
        # The subprocess handles its own timeout and exits cleanly.
        # We only use p.terminate() (SIGTERM, not SIGKILL) as a last resort.
        import multiprocessing as _mp

        _EXEC_TIMEOUT = min(timeout, 1000)

        def _worker_fn(code_str, result_queue, exec_timeout):
            """Run code in a subprocess. signal.alarm works here (main thread)."""
            import signal as _sig
            import sys as _sys
            import os as _os

            class _AlarmTimeout(BaseException):
                pass

            def _alarm_handler(signum, frame):
                raise _AlarmTimeout()

            _sig.signal(_sig.SIGALRM, _alarm_handler)
            _sig.alarm(exec_timeout)

            try:
                import numpy, math, random
                ns = {
                    "__builtins__": __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__).copy(),
                    "np": numpy, "numpy": numpy, "math": math, "random": random,
                    "evaluate_erdos_solution": lambda h, c, n: float(c),
                }
                exec(compile(code_str, "<llm>", "exec"), ns)
                if "run" not in ns:
                    result_queue.put({"error": "No 'run' function defined"})
                    return
                out = ns["run"](seed=42, budget_s=exec_timeout - 10)
                # Serialize result (numpy arrays can't cross process boundary directly)
                h = out[0].tolist() if hasattr(out[0], "tolist") else list(out[0])
                result_queue.put({"result": (h, float(out[1]), int(out[2]))})
            except _AlarmTimeout:
                result_queue.put({"error": f"Execution timed out after {exec_timeout}s"})
            except Exception as e:
                result_queue.put({"error": f"{type(e).__name__}: {str(e)[:300]}"})
            finally:
                _sig.alarm(0)  # Cancel any pending alarm

        q = _mp.Queue()
        p = _mp.Process(target=_worker_fn, args=(code, q, _EXEC_TIMEOUT))
        p.start()
        # Wait for subprocess: alarm should fire inside it, so give extra grace
        p.join(timeout=_EXEC_TIMEOUT + 30)

        if p.is_alive():
            # Subprocess didn't exit cleanly — send SIGTERM first (graceful)
            p.terminate()
            p.join(timeout=10)
            if p.is_alive():
                p.kill()  # Last resort
                p.join(timeout=5)
            return {
                "reward": 0.0, "raw_score": None,
                "error_msg": f"Subprocess terminated after {_EXEC_TIMEOUT}s",
                "stdout": "".join(stdout_capture),
            }

        if q.empty():
            return {
                "reward": 0.0, "raw_score": None,
                "error_msg": "Subprocess exited without result",
                "stdout": "".join(stdout_capture),
            }

        try:
            exec_result = q.get_nowait()
        except Exception:
            return {
                "reward": 0.0, "raw_score": None,
                "error_msg": "Failed to read subprocess result",
                "stdout": "".join(stdout_capture),
            }

        if "error" in exec_result:
            return {
                "reward": 0.0, "raw_score": None,
                "error_msg": exec_result["error"],
                "stdout": "".join(stdout_capture),
            }

        raw = exec_result["result"]
        h_values = np.asarray(raw[0], dtype=np.float64)
        c5_bound = raw[1]
        n_points = raw[2]
        result = (h_values, c5_bound, n_points)



        # Verify
        computed_c5 = verify_c5_solution(h_values, c5_bound, n_points)

        if computed_c5 <= 0 or not np.isfinite(computed_c5):
            return {
                "reward": 0.0, "raw_score": None,
                "error_msg": f"Invalid C5: {computed_c5}",
                "stdout": "".join(stdout_capture),
            }

        return {
            "reward": float(1.0 / (1e-8 + computed_c5)),
            "raw_score": float(computed_c5),
            "error_msg": "",
            "result_construction": h_values.tolist(),
            "n_points": int(n_points),
            "stdout": "".join(stdout_capture),
        }

    except Exception as e:
        return {
            "reward": 0.0, "raw_score": None,
            "error_msg": f"{type(e).__name__}: {str(e)[:300]}",
            "stdout": "".join(stdout_capture),
        }


def _extract_code(response: str) -> str:
    """Extract Python code from LLM response."""
    code_re = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
    blocks = code_re.findall(response)
    if blocks:
        return blocks[-1].strip()
    # If no code block, try the whole response
    return response.strip()


# ═══════════════════════════════════════════════════════════════════
# Initial state generation (from reference)
# ═══════════════════════════════════════════════════════════════════

def create_initial_state(rng=None):
    """Create a random initial state — matches reference exactly."""
    if rng is None:
        rng = np.random.default_rng()
    n_points = int(rng.integers(40, 100))
    construction = np.ones(n_points) * 0.5
    perturbation = rng.uniform(-0.4, 0.4, n_points)
    perturbation = perturbation - np.mean(perturbation)
    construction = construction + perturbation
    dx = 2.0 / n_points
    correlation = np.correlate(construction, 1 - construction, mode="full") * dx
    c5_bound = float(np.max(correlation))
    return {
        "construction": construction.tolist(),
        "c5_bound": c5_bound,
        "n_points": n_points,
        "code": "",
        "parent_c5": None,
        "observation": "",
    }


def erdos_ref_state_from_create_initial(rng: np.random.Generator) -> ErdosRefState:
    """Match ttt-discover-ref ErdosMinOverlapEnv.create_initial_state → State."""
    d = create_initial_state(rng)
    return ErdosRefState(
        timestep=-1,
        construction=list(d["construction"]),
        code="",
        value=-float(d["c5_bound"]),
        parent_values=[],
        parents=[],
        observation="",
    )


# ═══════════════════════════════════════════════════════════════════
# Prompt construction (from reference State.to_prompt + ErdosMinOverlapEnv.get_question)
# ═══════════════════════════════════════════════════════════════════

TARGET_C5 = 0.3808


def state_to_prompt(state: dict) -> str:
    """Build the state context portion — matches reference State.to_prompt()."""
    c5 = state.get("c5_bound", None)
    parent_c5 = state.get("parent_c5", None)
    code = state.get("code", "")
    observation = state.get("observation", "")

    value_ctx = "You are iteratively optimizing C₅ bound."

    if code and code.strip():
        value_ctx += f"\nHere is the last code we ran:\n```python\n{code}\n```"
    else:
        value_ctx += "\nNo previous code available."

    if parent_c5 is not None and c5 is not None:
        current_gap = c5 - TARGET_C5
        value_ctx += (
            f"\nHere is the C₅ bound before and after running the code above "
            f"(lower is better): {parent_c5:.6f} -> {c5:.6f}"
        )
        value_ctx += (
            f"\nTarget: {TARGET_C5}. Current gap: {current_gap:.6f}. "
            f"Further improvements will also be generously rewarded."
        )
    elif c5 is not None:
        current_gap = c5 - TARGET_C5
        value_ctx += f"\nCurrent C₅ bound (lower is better): {c5:.6f}"
        value_ctx += (
            f"\nTarget: {TARGET_C5}. Current gap: {current_gap:.6f}. "
            f"Further improvements will also be generously rewarded."
        )
    else:
        value_ctx += f"\nTarget C₅ bound: {TARGET_C5}"

    if observation and observation.strip():
        stdout = observation.strip()
        if len(stdout) > 500:
            stdout = "\n\n\t\t ...(TRUNCATED)...\n" + stdout[-500:]
        value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"

    return value_ctx


def build_erdos_question(state: dict) -> str:
    """Build the full Erdős question — matches reference ErdosMinOverlapEnv.get_question()."""
    state_ctx = state_to_prompt(state)

    construction = state.get("construction", [])
    n = len(construction) if construction else 0

    construction_section = ""
    if construction and n > 0:
        construction_section = (
            f"\nYou may want to start your search from the current construction, "
            f"which you can access through the `initial_h_values` global variable "
            f"(n={n} samples).\n"
            f"You are encouraged to explore solutions that use other starting points "
            f"to prevent getting stuck in a local optimum.\n"
        )

    code = state.get("code", "")
    if code and code.strip():
        code_section = (
            "Reason about how you could further improve this construction.\n"
            "Ideally, try to do something different than the above algorithm. "
            "Could be using different algorithmic ideas, adjusting your heuristics, "
            "adjusting / sweeping your hyperparemeters, etc.\n"
            "Unless you make a meaningful improvement, you will not be rewarded."
        )
    else:
        code_section = "Write code to optimize this construction."

    return f"""You are an expert in harmonic analysis, numerical optimization, and mathematical discovery.
Your task is to find an improved upper bound for the Erdős minimum overlap problem constant C₅.

## Problem

Find a step function h: [0, 2] → [0, 1] that **minimizes** the overlap integral:

$$C_5 = \\max_k \\int h(x)(1 - h(x+k)) dx$$

**Constraints**:
1. h(x) ∈ [0, 1] for all x
2. ∫₀² h(x) dx = 1

**Discretization**: Represent h as n_points samples over [0, 2].
With dx = 2.0 / n_points:
- 0 ≤ h[i] ≤ 1 for all i
- sum(h) * dx = 1 (equivalently: sum(h) == n_points / 2 exactly)

The evaluation computes: C₅ = max(np.correlate(h, 1-h, mode="full") * dx)

Smaller sequences with less than 1k samples are preferred - they are faster to optimize and evaluate.

**Lower C₅ values are better** - they provide tighter upper bounds on the Erdős constant.

## Budget & Resources
- **Time budget**: 1000s for your code to run
- **CPUs**: 2 available

## Rules
- Define `run(seed=42, budget_s=1000, **kwargs)` that returns `(h_values, c5_bound, n_points)`
- Use scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math
- Make all helper functions top level, no closures or lambdas
- No filesystem or network IO
- `evaluate_erdos_solution()` and `initial_h_values` (an initial construction, if available) are pre-imported
- Your function must complete within budget_s seconds and return the best solution found

**Lower is better**. Current record: C₅ ≤ 0.38092. Our goal is to find a construction that shows C₅ ≤ 0.38080.

{state_ctx}
{construction_section}
{code_section}
"""


# ═══════════════════════════════════════════════════════════════════
# NeMo RL Environment
# ═══════════════════════════════════════════════════════════════════

ErdosMetadata = dict[str, Any]


@ray.remote
class ErdosDiscoveryEnvironment(EnvironmentInterface):
    """Erdős Minimum Overlap environment for NeMo RL GRPO.

    Matches the reference TTT-Discover implementation:
    - C5 formulation with np.correlate
    - run() function entrypoint
    - scipy/cvxpy allowed
    - State context with parent code + improvement tracking
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self.sandbox_timeout = config.get("sandbox_timeout", 1000)
        self.num_initial_states = int(config.get("num_initial_states", 8))
        self.puct_c = float(config.get("puct_c", 1.0))
        self.puct_seed_batch_size = int(
            config.get("puct_seed_batch_size", self.num_initial_states)
        )

        # Tracking
        self.best_reward = 0.0
        self.best_c5 = float("inf")
        self.total_verified = 0
        self.total_valid = 0

        log_dir = config.get("puct_log_dir") or os.environ.get(
            "ERDOS_PUCT_LOG_DIR", "/tmp/erdos_puct"
        )
        os.makedirs(log_dir, exist_ok=True)
        sampler_path = os.path.join(log_dir, "puct_sampler.json")
        resume_step = config.get("puct_resume_step")
        if resume_step is not None:
            resume_step = int(resume_step)

        self.sampler = ErdosRefPUCTSampler(
            file_path=sampler_path,
            init_state_fn=lambda: erdos_ref_state_from_create_initial(
                np.random.default_rng()
            ),
            max_buffer_size=int(config.get("puct_max_buffer_size", 1000)),
            batch_size=self.puct_seed_batch_size,
            resume_step=resume_step,
            puct_c=self.puct_c,
            topk_children=int(config.get("puct_topk_children", 2)),
            max_construction_len=int(config.get("puct_max_construction_len", 1000)),
        )

    def get_initial_states(self, n: int = None) -> list[dict]:
        """Random states (e.g. validation). Training uses puct_sample_states()."""
        rng = np.random.default_rng(42)
        k = n if n is not None else self.num_initial_states
        return [create_initial_state(rng) for _ in range(k)]

    def puct_sample_states(self, num_prompts: int) -> list[dict]:
        """ttt-discover-ref PUCTSampler.sample_states — prompts + serial parent State."""
        picked = self.sampler.sample_states(num_prompts)
        out: list[dict] = []
        for s in picked:
            prompt_state = erdos_ref_state_to_prompt_state(s)
            prompt_state["erdos_ref_state"] = s.to_dict()
            out.append(prompt_state)
        return out

    def step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[ErdosMetadata],
    ) -> EnvironmentReturn[ErdosMetadata]:
        """Evaluate a batch of LLM responses."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        # Always use sync path (Ray actors run in event loops)
        return self._sync_step(message_log_batch, metadata)

    def _sync_step(
        self,
        message_log_batch: list[LLMMessageLogType],
        metadata: list[ErdosMetadata],
    ) -> EnvironmentReturn[ErdosMetadata]:
        """Synchronous step — executes code and computes C5 reward."""
        batch_size = len(message_log_batch)
        rewards = torch.zeros(batch_size)
        terminateds = torch.ones(batch_size)
        observations = [{"role": "user", "content": ""} for _ in range(batch_size)]
        answers = [None] * batch_size
        updated_metadata = list(metadata)

        import time as _time
        _t0 = _time.time()
        step_num = getattr(self, "_step_count", 0) + 1
        self._step_count = step_num
        print(f"[{_time.strftime('%H:%M:%S')}] 🧪 Starting reward computation for {batch_size} rollouts")

        for i, message_log in enumerate(message_log_batch):
            if i > 0 and i % 50 == 0:
                elapsed = _time.time() - _t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (batch_size - i) / rate if rate > 0 else 0
                print(
                    f"[{_time.strftime('%H:%M:%S')}] Reward progress: {i}/{batch_size} "
                    f"({elapsed:.0f}s elapsed, {rate:.1f} it/s, ~{eta:.0f}s remaining)"
                )

            # Extract assistant response
            response_text = ""
            for msg in reversed(message_log):
                if msg.get("role") == "assistant":
                    response_text = msg.get("content", "")
                    break

            # Extract code and execute
            code = _extract_code(response_text)

            # Inject initial_h_values from state if available
            state = metadata[i] if i < len(metadata) else {}
            construction = state.get("construction", None)
            parent_erdos: Optional[ErdosRefState] = None
            raw_parent = state.get("erdos_ref_state")
            if raw_parent is not None:
                try:
                    parent_erdos = ErdosRefState.from_dict(raw_parent)
                except Exception as e:
                    logger.warning("Invalid erdos_ref_state in metadata: %s", e)

            preamble = "import numpy as np\n\n"
            if construction:
                preamble += f"initial_h_values = np.array({construction!r})\n\n"
            else:
                preamble += "initial_h_values = None\n\n"

            full_code = preamble + code
            result = _execute_run_function(full_code, timeout=self.sandbox_timeout)

            reward = result.get("reward", 0.0)
            rewards[i] = reward
            self.total_verified += 1

            c5 = result.get("raw_score", None)

            if reward > 0:
                self.total_valid += 1
                if c5 is not None and c5 < self.best_c5:
                    self.best_c5 = c5
                    self.best_reward = reward
                    print(f"🏆 NEW BEST C5: {c5:.6f} (reward={reward:.4f})")

                answers[i] = f"C5={c5:.6f}" if c5 else f"reward={reward:.4f}"

            if parent_erdos is not None:
                if (
                    reward > 0
                    and c5 is not None
                    and result.get("result_construction") is not None
                ):
                    child = ErdosRefState(
                        timestep=step_num,
                        construction=list(result["result_construction"]),
                        code=code,
                        value=-float(c5),
                        observation=str(result.get("stdout", "") or ""),
                    )
                    try:
                        self.sampler.update_states(
                            [child], [parent_erdos], save=False
                        )
                    except Exception as e:
                        logger.warning("PUCT update_states failed: %s", e)
                else:
                    self.sampler.record_failed_rollout(parent_erdos)

            if i < len(updated_metadata):
                updated_metadata[i] = {
                    **updated_metadata[i],
                    "reward": reward,
                    "c5_bound": c5,
                    "error_msg": result.get("error_msg", ""),
                    "stdout": result.get("stdout", ""),
                    # Update state for PUCT if valid
                    "result_construction": result.get("result_construction"),
                    "n_points": result.get("n_points"),
                }

        elapsed = _time.time() - _t0
        valid = sum(1 for r in rewards if r > 0)
        max_r = float(rewards.max()) if len(rewards) > 0 else 0.0
        batch_best_c5 = 1.0 / max_r if max_r > 0 else float("inf")
        global_best = self.best_c5 if self.best_c5 < float("inf") else float("inf")
        print(
            f"\n{'='*60}\n"
            f"🎯 STEP REWARDS: {batch_size} rollouts in {elapsed:.1f}s\n"
            f"   valid:        {valid}/{batch_size} ({100*valid/max(1,batch_size):.1f}%)\n"
            f"   avg_reward:   {sum(float(r) for r in rewards)/max(1,batch_size):.6f}\n"
            f"   max_reward:   {max_r:.6f}\n"
            f"   batch_best_C5: {batch_best_c5:.6f}\n"
            f"   GLOBAL_BEST_C5: {global_best:.6f}\n"
            f"{'='*60}"
        )

        # Save outputs to JSONL for debugging
        import json

        log_dir = os.environ.get("ERDOS_LOG_DIR", "/tmp/erdos_outputs")
        os.makedirs(log_dir, exist_ok=True)
        out_path = os.path.join(log_dir, f"step_{step_num:03d}.jsonl")
        try:
            with open(out_path, "w") as fout:
                # Save a sample: first 10 + all valid ones
                for idx in range(batch_size):
                    r = float(rewards[idx])
                    save_this = (idx < 10) or (r > 0)
                    if not save_this:
                        continue
                    response = ""
                    for msg in reversed(message_log_batch[idx]):
                        if msg.get("role") == "assistant":
                            response = msg.get("content", "")
                            break
                    meta = updated_metadata[idx] if idx < len(updated_metadata) else {}
                    entry = {
                        "idx": idx,
                        "reward": r,
                        "c5_bound": meta.get("c5_bound"),
                        "error_msg": meta.get("error_msg", ""),
                        "response_len": len(response),
                        "response_preview": response[:500],
                        "code_preview": _extract_code(response)[:500] if response else "",
                    }
                    fout.write(json.dumps(entry) + "\n")
            print(f"   📝 Saved outputs to {out_path}")
        except Exception as e:
            print(f"   ⚠️ Failed to save outputs: {e}")

        try:
            self.sampler.flush(step_num)
        except Exception as e:
            logger.warning("PUCT flush failed: %s", e)

        return EnvironmentReturn(
            observations=observations,
            metadata=updated_metadata,
            next_stop_strings=[None] * batch_size,
            rewards=rewards,
            terminateds=terminateds,
            answers=answers,
        )

    def global_post_process_and_metrics(
        self,
        metadata: list[ErdosMetadata],
    ) -> tuple[list[ErdosMetadata], dict[str, float]]:
        """Compute aggregate metrics after a step."""
        metrics = {
            "best_c5": self.best_c5 if self.best_c5 < float("inf") else 0.0,
            "best_reward": self.best_reward,
            "total_verified": float(self.total_verified),
            "total_valid": float(self.total_valid),
            "valid_rate": (
                self.total_valid / max(1, self.total_verified)
            ),
        }

        # Count valid solutions in this batch
        batch_rewards = [m.get("reward", 0.0) for m in metadata]
        batch_valid = sum(1 for r in batch_rewards if r > 0)
        batch_c5s = [m.get("c5_bound") for m in metadata if m.get("c5_bound") is not None]

        metrics["erdos/max_reward"] = float(max(batch_rewards)) if batch_rewards else 0.0
        metrics["erdos/avg_reward"] = float(sum(batch_rewards) / max(1, len(batch_rewards)))
        metrics["erdos/valid_count"] = float(batch_valid)
        metrics["erdos/valid_rate"] = float(batch_valid / max(1, len(batch_rewards)))
        metrics["erdos/batch_size"] = float(len(metadata))

        if batch_c5s:
            metrics["erdos/best_c5"] = float(min(batch_c5s))
            metrics["erdos/mean_c5"] = float(sum(batch_c5s) / len(batch_c5s))
            metrics["erdos/worst_c5"] = float(max(batch_c5s))

        metrics["erdos/global_best_c5"] = float(self.best_c5) if self.best_c5 < float("inf") else 0.0
        metrics["erdos/global_valid_total"] = float(self.total_valid)

        # Print summary to driver log
        max_r = metrics["erdos/max_reward"]
        avg_r = metrics["erdos/avg_reward"]
        best = metrics.get("erdos/best_c5", "n/a")
        print(f"  🎯 Erdős: avg_reward={avg_r:.4f} max_reward={max_r:.4f} "
              f"valid={batch_valid}/{len(metadata)} "
              f"best_c5={best}")

        try:
            for k, v in self.sampler.get_sample_stats().items():
                metrics[f"erdos/{k}"] = float(v)
        except Exception:
            pass

        return metadata, metrics
