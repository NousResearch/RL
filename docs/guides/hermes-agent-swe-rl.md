# Hermes Agent SWE-RL Training

Train [hermes-agent](https://github.com/NousResearch/hermes-agent) on SWE-bench tasks using NeMo RL's GRPO pipeline. This replaces the OpenHands agent in the Stage 2.2 SWE-RL pipeline with hermes-agent, preserving the full hermes toolset (terminal, file operations, search, patch, etc.) and training on the model's actual production behavior.

## Overview

The hermes agent runs inside Apptainer SWE-bench sandbox containers, solving code bugs using its standard tool-calling interface. Each rollout produces a multi-turn trajectory with per-turn token IDs and logprobs, which NeMo RL uses for GRPO training.

```
NeMo RL GRPO Loop
├── Megatron Policy (training)
├── vLLM Inference (rollout generation via NeMo Gym VLLMModel proxy)
└── NeMo Gym SWE Agent Server
    └── RunHermesAgent → Apptainer SWE-bench container
        ├── /testbed (repo at base_commit)
        ├── /opt/hermes-agent (built at container startup)
        └── hermes_runner.py → AIAgent.run_conversation()
            ├── Full hermes toolset (16 tools)
            ├── Token IDs + logprobs captured per-turn
            └── git diff → SWE-bench eval → binary reward
```

## Requirements

- **Container**: `nvcr.io/nvidia/nemo-rl:v0.5.0.nemotron_3_super` (official NGC)
  - Must use the official container — manually-built containers may lack tool parser support (`qwen3_coder`)
- **Model**: Any model with tool-calling support. Tested with `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- **Gym**: [NousResearch/Gym](https://github.com/NousResearch/Gym) branch `add-hermes-agent-to-super-training`
- **SIF Images**: Apptainer `.sif` files for SWE-bench instances
- **Cluster**: Slurm with Pyxis/Enroot, 8+ GPU nodes

## Quick Start

### 1. Clone repos

```bash
git clone --branch super-v3-hermes-agent https://github.com/NousResearch/RL.git
cd RL
# Symlink the hermes Gym branch
git clone --branch add-hermes-agent-to-super-training https://github.com/NousResearch/Gym.git
ln -sfn $(pwd)/../Gym 3rdparty/Gym-workspace/Gym
git submodule update --init 3rdparty/Megatron-LM-workspace/Megatron-LM 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
```

### 2. Download model + SIF images

```bash
# Model
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --local-dir models/nano

# SIF images (for full SWE-bench training)
./examples/nemo_gym/download_swe_images.py --sif-dir /path/to/sif --concurrency 16
```

### 3. Prepare data

Training data must be JSONL in NeMo Gym SWE-bench format:

```json
{
  "agent_ref": {"name": "swe_agents_train", "type": "responses_api_agents"},
  "responses_create_params": {
    "input": [{"role": "user", "content": "...problem statement..."}],
    "model": "/path/to/model",
    "metadata": {
      "instance_id": "repo__project-1234",
      "problem_statement": "...",
      "instance_dict": "{...serialized SWE-bench instance...}",
      ...
    }
  }
}
```

Key fields:
- `agent_ref.name` must match config (`swe_agents_train` / `swe_agents_val`)
- `metadata.instance_dict` is a JSON-serialized string
- `input` must contain at least one user message
- `model` must match the vLLM served model name (full path)

### 4. Launch

See `examples/configs/super/stage2_hermes_nano_8node.yaml` for the full config. Launch via `super_launch.sh` or directly:

```bash
CONTAINER="nvcr.io#nvidia/nemo-rl:v0.5.0.nemotron_3_super"
# ... set MOUNTS, COMMAND, SETUP_COMMAND ...
sbatch --nodes=8 ray.sub
```

The SETUP_COMMAND should:
1. Install apptainer
2. Install `protobuf grpcio yappi` (missing from base container)
3. Prefetch Gym venvs at `/opt/gym_venvs`
4. Clone + build hermes-agent at `/opt/hermes-agent`

## Architecture (8-node config)

| Component | Nodes | GPUs | Config |
|-----------|-------|------|--------|
| vLLM Inference | 4 | 32 | TP=4, 8 workers, `qwen3_coder` tool parser |
| Megatron Training | 4 | 32 | TP=2, EP=8, PP=1, CP=1, DP=2 |
| NeMo Gym | (on inference nodes) | 0 | SWE agent servers |

## Changes from Upstream

### NeMo RL (this repo)
- `examples/configs/super/stage2_hermes_nano_8node.yaml` — 8-node config for Nano with hermes
- `nemo_rl/models/generation/vllm/vllm_worker_async.py` — Override temperature/top_p instead of asserting (hermes requests may not match generation config exactly)
- `ray.sub` — Cluster-specific patches (pmi2, container-writable, NCCL IB, ssh coordination)

### NeMo Gym ([NousResearch/Gym](https://github.com/NousResearch/Gym/tree/add-hermes-agent-to-super-training))
- `run_hermes.py` — `RunHermesAgent` class + in-container runner script
- `utils.py` — Treat hermes trajectory as OpenHands format for token ID extraction
- `configs/swebench_hermes_training.yaml` — Train/val config for hermes framework
- Removed `--pid` from Apptainer exec (breaks in nested Pyxis containers)

## Known Issues

- **Agent rollouts are slow**: Each rollout takes 5-10 minutes with thinking mode. This is expected for multi-turn agent interactions.
- **Container version matters**: The official NGC container has tool parsers. Manually-built containers may lack them (empty `ToolParserManager.tool_parsers`).
- **Bad GPU nodes**: Some cluster nodes have CUDA errors. Use `--exclude` in sbatch to skip them.
- **Gym venvs**: The base container doesn't have pre-baked SWE venvs. Build them via SETUP_COMMAND using `prefetch_venvs.py` or install deps system-wide.
