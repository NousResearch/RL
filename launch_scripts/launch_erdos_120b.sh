#!/bin/bash
# TTT-Discover Erdős — Nemotron-3-Super-120B, 8k seq len, wandb logging
set -euo pipefail
cd /home/mormio/RL

CONTAINER="/home/shared/containers/nemo-rl-super-v3.sqsh"
MODEL_PATH="/home/shared/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16"
EXP="results/erdos-120b-$(date +%Y%m%d_%H%M)"
mkdir -p "$EXP"

WANDB_API_KEY=$(grep 'password' ~/.netrc | head -1 | awk '{print $2}')
MOUNTS="$PWD:$PWD,$MODEL_PATH:$MODEL_PATH,$HOME/.cache:$HOME/.cache"

COMMAND="
cd /opt/nemo-rl && \
export NCCL_BUFFSIZE=33554432 && \
export CUDA_DEVICE_ORDER=PCI_BUS_ID && \
export NCCL_IB_AR_THRESHOLD=0 && \
export NCCL_IB_PCI_RELAXED_ORDERING=1 && \
export NCCL_IB_QPS_PER_CONNECTION=2 && \
export NCCL_IB_SPLIT_DATA_ON_QPS=0 && \
export NCCL_IGNORE_CPU_AFFINITY=1 && \
export NCCL_IB_HCA=mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_13:1,mlx5_14:1,mlx5_15:1 && \
export NCCL_SOCKET_IFNAME=bond0 && \
export UCX_NET_DEVICES=bond0 && \
export HF_HUB_ENABLE_HF_TRANSFER=0 && \
export TORCH_CUDA_ARCH_LIST='9.0 10.0' && \
export NRL_IGNORE_VERSION_MISMATCH=1 && \
export ERDOS_LOG_DIR=/home/mormio/RL/results/erdos_outputs && \
export ERDOS_PUCT_LOG_DIR=/home/mormio/RL/results/erdos_puct && \
export WANDB_API_KEY=$WANDB_API_KEY && \

SRC=/home/mormio/RL
cp \$SRC/nemo_rl/algorithms/entropic_advantage_estimator.py /opt/nemo-rl/nemo_rl/algorithms/
cp \$SRC/nemo_rl/environments/erdos_discovery_environment.py /opt/nemo-rl/nemo_rl/environments/
cp \$SRC/nemo_rl/environments/erdos_ref_puct_sampler.py /opt/nemo-rl/nemo_rl/environments/
cp \$SRC/examples/run_discover.py /opt/nemo-rl/examples/
cp \$SRC/examples/configs/grpo_erdos_discover.yaml /opt/nemo-rl/examples/configs/

python -c \"
path = '/opt/nemo-rl/nemo_rl/algorithms/grpo.py'
with open(path) as f:
    content = f.read()
if 'entropic_adaptive_beta' not in content:
    old = '    else:\\n        raise ValueError(f\\\"Invalid adv_estimator name: {adv_estimator_name}\\\")\\n\\n    return adv_estimator'
    new = '''    elif adv_estimator_name == \\\"entropic_adaptive_beta\\\":
        from nemo_rl.algorithms.entropic_advantage_estimator import (
            EntropicAdaptiveBetaAdvantageEstimator,
        )
        adv_estimator = EntropicAdaptiveBetaAdvantageEstimator(
            adv_estimator_config, loss_config
        )
        print(\\\"  Using Entropic Adaptive-Beta advantage estimator (TTT-Discover)\\\")
    else:
        raise ValueError(f\\\"Invalid adv_estimator name: {adv_estimator_name}\\\")

    return adv_estimator'''
    content = content.replace(old, new)
    with open(path, 'w') as f:
        f.write(content)
    print('Patched grpo.py')
\" && \

python -c \"
path = '/opt/nemo-rl/nemo_rl/environments/utils.py'
with open(path) as f:
    content = f.read()
if 'erdos_discovery' not in content:
    content = content.replace(
        '\\\"nemo_gym\\\": {',
        '\\\"erdos_discovery\\\": {\\n        \\\"actor_class_fqn\\\": \\\"nemo_rl.environments.erdos_discovery_environment.ErdosDiscoveryEnvironment\\\",\\n    },\\n    \\\"nemo_gym\\\": {'
    )
    with open(path, 'w') as f:
        f.write(content)
    print('Patched utils.py')
\" && \

python examples/run_discover.py \
  --config examples/configs/grpo_erdos_discover.yaml
"

echo "Submitting Erdős TTT-Discover 120B (8k seq, wandb)..."
echo "  Container: $CONTAINER"
echo "  Model:     $MODEL_PATH"
echo "  Nodes:     8 (2 inference + 6 training)"
echo "  Seq len:   16384"
echo "  Exp:       $EXP"

COMMAND="$COMMAND" \
CONTAINER="$CONTAINER" \
MOUNTS="$MOUNTS" \
GPUS_PER_NODE=8 \
sbatch \
  --nodes=8 --partition=batch --exclusive \
  --job-name=erdos-120b --time=12:00:00 \
  --output="$EXP/slurm-%j.out" \
  --error="$EXP/slurm-%j.err" \
  --exclude=d2dfac12-001,d2dfac12-002,d2dfac12-004,d2dfac12-007,d2dfac12-008,d2dfac12-019,d2dfac12-027,d2dfac12-028,d2dfac12-029 \
  ray.sub

echo "Logs: $EXP/"
echo "W&B: https://wandb.ai/nous_research/ttt-discover-erdos"
