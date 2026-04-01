"""Run script for TTT-Discover GRPO training on the Erdős Minimum Overlap Problem.

This follows the sliding_puzzle pattern: custom IterableDataset that generates
prompts dynamically from a PUCT buffer, wired into the standard GRPO loop.

Usage:
    # Start the Gym resource server first (separate process/node):
    cd ~/Gym && ng_run "+config_paths=[resources_servers/erdos_discovery/configs/erdos_discovery.yaml]"

    # Then run training:
    cd ~/RL && uv run python examples/run_discover.py [--config examples/configs/grpo_erdos_discover.yaml]

Reference: "Learning to Discover at Test Time" (arXiv:2601.16175)
"""

import itertools
import argparse
import itertools
import logging
import os
import sys
from typing import Optional

import aiohttp
import asyncio
import numpy as np
import ray
import torch
from torch.utils.data import IterableDataset

from nemo_rl.algorithms.grpo import MasterConfig, grpo_train, setup
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data.interfaces import DatumSpec, LLMMessageLogType
from nemo_rl.distributed.virtual_cluster import init_ray
from nemo_rl.environments.erdos_discovery_environment import (
    ErdosDiscoveryEnvironment,
)
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.utils.config import load_config

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Problem description (same as in the Gym resource server)
# ═══════════════════════════════════════════════════════════════════

PROBLEM_DESCRIPTION = """\
Erdos Minimum Overlap Problem
==============================

Goal: Find a step function f (Python list or NumPy array) giving the
tightest possible upper bound on the Erdos minimum overlap constant c.

Background:
  For integer n, partition {1,...,2n} into equal sets A, B.
  M_k = #{(a,b) : a in A, b in B, a-b=k}.
  c = lim_{n->inf} min_{A,B} max_k M_k / n.

Known bounds: 0.379005 < c < 0.380927 (Haugland 2016)
Current best upper bound: 0.380876 (2026)

Upper Bound via Step Functions:
  f : [0,1] -> [0,1] with mean(f) = 0.5 gives:
    bound = 2*n*max(autocorr(f)) / sum(f)^2
  where autocorr is computed via FFT.
  Smaller bound -> higher reward (reward = 1/bound).

Constraints: 1 <= len(f) <= 1000, 0 <= f[i] <= 1, mean(f) ~ 0.5 (tol 1e-3).

Output: Python code defining variable `f` in a ```python block.
Allowed: numpy, math, random, itertools, functools, collections.
Execution limit: 600 seconds. Target: bound < 0.380876.\
"""


# ═══════════════════════════════════════════════════════════════════
# Datum generation
# ═══════════════════════════════════════════════════════════════════


def generate_discover_datum(
    tokenizer,
    state_info: dict,
    idx: int,
    task_name: str = "erdos_discovery",
) -> DatumSpec:
    """Create a DatumSpec from a PUCT-selected state.

    Args:
        tokenizer: HuggingFace tokenizer.
        state_info: Dict from /select_state with keys:
            state, context, reward, system_prompt, user_prompt.
        idx: Datum index.
        task_name: Task name for env routing.

    Returns:
        DatumSpec ready for the GRPO training loop.
    """
    system_prompt = state_info.get("system_prompt", PROBLEM_DESCRIPTION)
    user_prompt = state_info["user_prompt"]

    messages: LLMMessageLogType = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Tokenize the prompt
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long)

    # Attach token_ids to messages for NeMo RL's message_log format
    for msg in messages:
        msg_text = tokenizer.apply_chat_template(
            [msg], tokenize=False, add_generation_prompt=False
        )
        msg_ids = tokenizer.encode(msg_text, add_special_tokens=False)
        msg["token_ids"] = torch.tensor(msg_ids, dtype=torch.long)

    return DatumSpec(
        message_log=messages,
        length=len(prompt_ids),
        extra_env_info={
            "parent_state": state_info.get("state"),
            "context": state_info.get("context"),
            "reward": state_info.get("reward", 0.0),
        },
        loss_multiplier=1.0,
        idx=idx,
        task_name=task_name,
    )


# ═══════════════════════════════════════════════════════════════════
# Dynamic dataset backed by PUCT buffer
# ═══════════════════════════════════════════════════════════════════


class DiscoverDataset(IterableDataset):
    """Iterable dataset that fetches prompts from the PUCT buffer each step.

    Each iteration fetches `num_groups_per_step` states from the Gym resource
    server's /select_state endpoint and yields them as DatumSpecs.

    The dataset loops indefinitely — the training loop controls termination
    via max_num_steps in the GRPO config.
    """

    def __init__(
        self,
        tokenizer,
        resource_server_url: str,
        num_groups_per_step: int = 8,
        task_name: str = "erdos_discovery",
        length: int = 1000,  # Nominal length for dataloader
    ):
        self.tokenizer = tokenizer
        self.resource_server_url = resource_server_url
        self.num_groups_per_step = num_groups_per_step
        self.task_name = task_name
        self.length = length
        self._idx_counter = itertools.count()

    def _fetch_states_sync(self) -> list[dict]:
        """Synchronously fetch states from the PUCT buffer."""
        import requests

        try:
            resp = requests.post(
                f"{self.resource_server_url}/select_state",
                json={
                    "batch_size": self.num_groups_per_step,
                    "num_groups": self.num_groups_per_step,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("states", [])
        except Exception as e:
            logger.error("Failed to fetch states from PUCT buffer: %s", e)
            # Return fallback: single default prompt
            return [
                {
                    "state": [0.5] * 50,
                    "context": [],
                    "reward": 0.5,
                    "system_prompt": PROBLEM_DESCRIPTION,
                    "user_prompt": (
                        "Starting construction (bound=2.000000, 50 pieces):\n"
                        "[0.5000, 0.5000, ..., 0.5000]\n\n"
                        "Improve on this construction. Write Python code that "
                        "defines a better step function `f`. Think carefully."
                    ),
                }
            ]

    def __iter__(self):
        for _ in itertools.count():
            states = self._fetch_states_sync()
            for state_info in states:
                idx = next(self._idx_counter)
                yield generate_discover_datum(
                    self.tokenizer,
                    state_info,
                    idx=idx,
                    task_name=self.task_name,
                )

    def __len__(self):
        return self.length


# ═══════════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════════


def setup_discover_data(config: MasterConfig, tokenizer):
    """Create dataset, environment, and wire them together.

    Returns:
        (train_dataset, val_dataset, task_to_env, val_task_to_env)
    """
    env_config = config.get("env", {}).get("erdos_discovery", {})
    resource_server_url = env_config.get(
        "resource_server_url", "http://localhost:8080"
    )
    num_groups_per_step = env_config.get("num_groups_per_step", 8)
    task_name = "erdos_discovery"

    # Create the dynamic dataset
    train_dataset = DiscoverDataset(
        tokenizer=tokenizer,
        resource_server_url=resource_server_url,
        num_groups_per_step=num_groups_per_step,
        task_name=task_name,
        length=config["grpo"]["max_num_steps"] * num_groups_per_step,
    )

    # Validation dataset: same thing (could be a fixed set, but for discovery
    # we just re-sample from the buffer)
    val_dataset = DiscoverDataset(
        tokenizer=tokenizer,
        resource_server_url=resource_server_url,
        num_groups_per_step=num_groups_per_step,
        task_name=task_name,
        length=num_groups_per_step,
    )

    # Create the environment as a Ray actor
    env = ErdosDiscoveryEnvironment.options(
        num_gpus=0,
        max_restarts=-1,
        max_task_retries=-1,
    ).remote(config=env_config)

    task_to_env = {task_name: env}
    val_task_to_env = {task_name: env}

    return train_dataset, val_dataset, task_to_env, val_task_to_env


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════


def main():
    import os
    from omegaconf import OmegaConf
    from nemo_rl.utils.config import load_config

    # Register custom resolvers needed by the base config
    if not OmegaConf.has_resolver("mul"):
        OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
    if not OmegaConf.has_resolver("div"):
        OmegaConf.register_new_resolver("div", lambda a, b: a // b)

    try:
        from nemo_rl.utils.config import register_omegaconf_resolvers
        register_omegaconf_resolvers()
    except ImportError:
        pass  # v0.5.0 container doesn't have this

    # Parse --config argument
    config_path = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith("--config="):
            config_path = arg.split("=", 1)[1]
        elif arg == "--config" and i < len(sys.argv) - 1:
            config_path = sys.argv[i + 1]
        elif not arg.startswith("--") and config_path is None:
            config_path = arg

    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "configs", "grpo_erdos_discover_debug.yaml"
        )

    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    # Resolve OmegaConf interpolations (e.g. ${policy.model_name})
    oc = OmegaConf.create(config)
    config = OmegaConf.to_container(oc, resolve=True)

    # Initialize Ray
    init_ray()
    set_seed(config.get("seed", 42))

    # Tokenizer
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])

    # Generation config
    config["policy"]["generation"] = configure_generation_config(
        config["policy"]["generation"], tokenizer
    )

    # Setup data + environment
    train_dataset, val_dataset, task_to_env, val_task_to_env = (
        setup_discover_data(config, tokenizer)
    )

    # Setup returns vary across container versions — unpack dynamically
    setup_result = setup(config, tokenizer, train_dataset, val_dataset)

    # Inspect the grpo_train signature to know what to pass
    import inspect
    train_sig = inspect.signature(grpo_train)
    train_params = list(train_sig.parameters.keys())
    print(f"  setup() returned {len(setup_result)} values")
    print(f"  grpo_train() expects {len(train_params)} params: {train_params[:5]}...")

    # The standard pattern: setup returns everything grpo_train needs,
    # except task_to_env and val_task_to_env which we provide.
    # Detect where to inject them based on parameter names.
    setup_list = list(setup_result)

    # Build kwargs for grpo_train by matching setup outputs + our envs
    # Common signatures:
    # v0.5.0: setup returns (policy, gen, dl, val_dl, tokenizer, loss, env, val_env, logger, ckpt, state, config)
    # super-v3: may return more
    # Strategy: pass setup outputs positionally, but swap in our envs
    if len(setup_list) == 12:
        # v0.5.0 style: already includes env placeholders at positions 6,7
        setup_list[6] = task_to_env
        setup_list[7] = val_task_to_env
        grpo_train(*setup_list)
    elif len(setup_list) == 10:
        # Older style without envs
        policy, policy_generation, dataloader, val_dataloader, tokenizer_out, loss_fn, nemo_logger, checkpointer, grpo_state, master_config = setup_list
        grpo_train(
            policy, policy_generation, dataloader, val_dataloader,
            tokenizer_out, loss_fn, task_to_env, val_task_to_env,
            nemo_logger, checkpointer, grpo_state, master_config,
        )
    else:
        # Unknown format — try passing everything with envs injected
        # Find the positions of env-like params in grpo_train signature
        env_idx = None
        for i, p in enumerate(train_params):
            if 'task_to_env' in p and 'val' not in p:
                env_idx = i
                break
        if env_idx is not None:
            # Insert our envs at the right position
            args = list(setup_list)
            args.insert(env_idx, task_to_env)
            args.insert(env_idx + 1, val_task_to_env)
            grpo_train(*args[:len(train_params)])
        else:
            print(f"WARNING: Could not determine grpo_train signature, trying positional")
            grpo_train(*setup_list)


if __name__ == "__main__":
    main()
