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

    class _Timeout(Exception):
        pass

    def _handler(s, f):
        raise _Timeout(f"Execution timed out after {timeout}s")

    try:
        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout)
        try:
            exec(compile(code, "<llm>", "exec"), namespace)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        if "run" not in namespace:
            return {
                "reward": 0.0, "raw_score": None,
                "error_msg": "No 'run' function defined",
                "stdout": "".join(stdout_capture),
            }

        # Call run()
        signal.signal(signal.SIGALRM, _handler)
        signal.alarm(timeout)
        try:
            result = namespace["run"](seed=42, budget_s=timeout)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        if not isinstance(result, tuple) or len(result) != 3:
            return {
                "reward": 0.0, "raw_score": None,
                "error_msg": f"run() must return (h_values, c5_bound, n_points), got {type(result)}",
                "stdout": "".join(stdout_capture),
            }

        h_values, c5_bound, n_points = result
        h_values = np.asarray(h_values, dtype=np.float64)

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

    except _Timeout as e:
        return {
            "reward": 0.0, "raw_score": None,
            "error_msg": str(e),
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
        self.num_initial_states = config.get("num_initial_states", 16)

        # Tracking
        self.best_reward = 0.0
        self.best_c5 = float("inf")
        self.total_verified = 0
        self.total_valid = 0

        # PUCT buffer for state management
        self._states = []
        self._initialize_states()

    def _initialize_states(self):
        """Generate initial random states."""
        rng = np.random.default_rng(42)
        for _ in range(self.num_initial_states):
            self._states.append(create_initial_state(rng))

    def get_initial_states(self, n: int = None) -> list[dict]:
        """Return initial states for prompt generation."""
        if n is None:
            return self._states
        return self._states[:n]

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

        for i, message_log in enumerate(message_log_batch):
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
                    logger.info(f"New best C5: {c5:.6f} (reward={reward:.4f})")

                answers[i] = f"C5={c5:.6f}" if c5 else f"reward={reward:.4f}"

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
        batch_valid = sum(1 for m in metadata if m.get("reward", 0) > 0)
        batch_c5s = [m.get("c5_bound") for m in metadata if m.get("c5_bound") is not None]
        if batch_c5s:
            metrics["batch_best_c5"] = min(batch_c5s)
            metrics["batch_mean_c5"] = sum(batch_c5s) / len(batch_c5s)
        metrics["batch_valid_count"] = float(batch_valid)
        metrics["batch_size"] = float(len(metadata))

        return metadata, metrics
