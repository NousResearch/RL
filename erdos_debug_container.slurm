#!/bin/bash
#SBATCH --job-name=erdos-debug
#SBATCH --output=logs/erdos-debug-%j.out
#SBATCH --error=logs/erdos-debug-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --container-image=nvcr.io#nvidia/nemo-rl:v0.5.0
#SBATCH --container-writable
#SBATCH --container-mounts=/home/mormio/RL:/home/mormio/RL,/home/shared/models:/home/shared/models
#SBATCH --exclude=d2dfac12-001,d2dfac12-002,d2dfac12-004,d2dfac12-007,d2dfac12-008,d2dfac12-019,d2dfac12-027,d2dfac12-028,d2dfac12-029

# TTT-Discover debug — single node, inside NeMo RL container.
# Skips ray.sub complexity. Starts Ray inline and runs training.

set -eo pipefail

echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L | wc -l)"
echo "Python: $(python --version)"
echo "Job: $SLURM_JOB_ID"

# NCCL config
export NCCL_IB_HCA=mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_13:1,mlx5_14:1,mlx5_15:1
export NCCL_SOCKET_IFNAME=bond0
export UCX_NET_DEVICES=bond0
export HF_HUB_ENABLE_HF_TRANSFER=0

cd /home/mormio/RL

# Run training directly — Ray will start automatically via init_ray()
uv run python examples/run_discover.py \
    --config examples/configs/grpo_erdos_discover_debug.yaml

echo "Training complete"
