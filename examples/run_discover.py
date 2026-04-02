"""Run script for TTT-Discover GRPO training on the Erdős Minimum Overlap Problem.

Matches the reference implementation at:
  https://github.com/test-time-training/discover/blob/main/examples/erdos_min_overlap/env.py

Usage (inside NeMo RL container):
  python examples/run_discover.py --config examples/configs/grpo_erdos_discover.yaml
"""

import itertools
import logging
import os
import sys
from typing import Optional

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
    build_erdos_question,
    create_initial_state,
)
from nemo_rl.models.generation import configure_generation_config

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Datum generation
# ═══════════════════════════════════════════════════════════════════


def generate_discover_datum(
    tokenizer,
    state: dict,
    idx: int,
    task_name: str = "erdos_discovery",
) -> DatumSpec:
    """Create a DatumSpec from a state dict.

    The prompt is built using the reference TTT-Discover get_question() format.
    """
    user_prompt = build_erdos_question(state)

    messages: LLMMessageLogType = [
        {"role": "user", "content": user_prompt},
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

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
            "construction": state.get("construction"),
            "c5_bound": state.get("c5_bound"),
            "n_points": state.get("n_points"),
            "code": state.get("code", ""),
            "parent_c5": state.get("parent_c5"),
            "observation": state.get("observation", ""),
        },
        loss_multiplier=1.0,
        idx=idx,
        task_name=task_name,
    )


# ═══════════════════════════════════════════════════════════════════
# Dataset backed by initial states (PUCT selection comes later)
# ═══════════════════════════════════════════════════════════════════


class DiscoverDataset(IterableDataset):
    """Dataset that generates prompts from Erdős initial states.

    Each iteration generates diverse initial states and yields them as
    DatumSpecs with the reference TTT-Discover prompt format.

    For now, initial states are random perturbations of h=0.5 (matching
    the reference). Future: PUCT buffer selects states based on prior
    discoveries.
    """

    def __init__(
        self,
        tokenizer,
        num_states_per_step: int = 8,
        task_name: str = "erdos_discovery",
        length: int = 1000,
        seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.num_states_per_step = num_states_per_step
        self.task_name = task_name
        self.length = length
        self._idx_counter = itertools.count()
        self._rng = np.random.default_rng(seed)

    def __iter__(self):
        for _ in itertools.count():
            # Generate fresh initial states each step
            for _ in range(self.num_states_per_step):
                state = create_initial_state(self._rng)
                idx = next(self._idx_counter)
                yield generate_discover_datum(
                    self.tokenizer,
                    state,
                    idx=idx,
                    task_name=self.task_name,
                )

    def __len__(self):
        return self.length


# ═══════════════════════════════════════════════════════════════════
# Setup
# ═══════════════════════════════════════════════════════════════════


def setup_discover_data(config: MasterConfig, tokenizer):
    """Create dataset, environment, and wire them together."""
    env_config = config.get("env", {}).get("erdos_discovery", {})
    num_states = config.get("grpo", {}).get("num_prompts_per_step", 8)
    task_name = "erdos_discovery"

    train_dataset = DiscoverDataset(
        tokenizer=tokenizer,
        num_states_per_step=num_states,
        task_name=task_name,
        length=config.get("grpo", {}).get("max_num_steps", 50) * num_states,
        seed=config.get("seed", 42),
    )

    val_dataset = DiscoverDataset(
        tokenizer=tokenizer,
        num_states_per_step=num_states,
        task_name=task_name,
        length=num_states,
        seed=config.get("seed", 42) + 1,
    )

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
    from omegaconf import OmegaConf
    from nemo_rl.utils.config import load_config

    # Register custom resolvers
    if not OmegaConf.has_resolver("mul"):
        OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
    if not OmegaConf.has_resolver("div"):
        OmegaConf.register_new_resolver("div", lambda a, b: a // b)

    try:
        from nemo_rl.utils.config import register_omegaconf_resolvers
        register_omegaconf_resolvers()
    except ImportError:
        pass

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
            os.path.dirname(__file__), "configs", "grpo_erdos_discover.yaml"
        )

    print(f"Loading config from: {config_path}")
    config = load_config(config_path)

    # Resolve OmegaConf interpolations
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

    # Ensure checkpointing config exists (some container versions require it)
    if "checkpointing" not in config:
        config["checkpointing"] = {
            "enabled": False,
            "checkpoint_dir": "results/erdos",
            "save_period": 999999,
            "checkpoint_must_save_by": None,
            "model_save_format": "safetensors",
            "save_consolidated": False,
            "metric_name": "total_reward/mean",
            "higher_is_better": True,
            "keep_top_k": 1000000,
        }
    elif config["checkpointing"].get("checkpoint_must_save_by") is None:
        config["checkpointing"]["checkpoint_must_save_by"] = None

    # Setup returns vary across container versions
    setup_result = setup(config, tokenizer, train_dataset, val_dataset)
    setup_list = list(setup_result)
    n = len(setup_list)
    print(f"  setup() returned {n} values")

    if n == 11:
        # super-v3 container
        (policy, policy_generation, _nemo_gym, _clusters,
         dataloader, val_dataloader, loss_fn,
         nemo_logger, checkpointer, grpo_state, master_config) = setup_list
        grpo_train(
            policy, policy_generation,
            dataloader, val_dataloader,
            tokenizer, loss_fn,
            task_to_env, val_task_to_env,
            nemo_logger, checkpointer,
            grpo_state, master_config,
        )
    elif n == 10:
        # v0.5.0 container
        (policy, policy_generation, dataloader, val_dataloader,
         loss_fn, nemo_logger, checkpointer, grpo_state,
         master_config, _extra) = setup_list
        grpo_train(
            policy, policy_generation,
            dataloader, val_dataloader,
            tokenizer, loss_fn,
            task_to_env, val_task_to_env,
            nemo_logger, checkpointer,
            grpo_state, master_config,
        )
    else:
        raise RuntimeError(f"Unexpected setup() return count: {n}")


if __name__ == "__main__":
    main()
