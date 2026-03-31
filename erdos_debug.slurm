#!/bin/bash
#SBATCH --job-name=erdos-debug
#SBATCH --output=logs/erdos-debug-%j.out
#SBATCH --error=logs/erdos-debug-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=64
#SBATCH --exclude=d2dfac12-001,d2dfac12-002,d2dfac12-004,d2dfac12-007,d2dfac12-008,d2dfac12-019,d2dfac12-027,d2dfac12-028,d2dfac12-029

# TTT-Discover debug run — single node, Qwen3-1.7B, inline reward, 5 steps.
set -eo pipefail

mkdir -p logs

echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L | wc -l)"
echo "Job ID: $SLURM_JOB_ID"

# No Gym server needed — using inline mode for debug.
# The environment computes rewards directly in-process.

cd /home/mormio/RL

echo "Starting TTT-Discover GRPO debug training (inline mode)..."
PATH=$HOME/.local/bin:$PATH uv run python examples/run_discover.py --config \
    examples/configs/grpo_erdos_discover_debug.yaml

echo "Training complete"
