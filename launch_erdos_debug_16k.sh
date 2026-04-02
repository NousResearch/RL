#!/bin/bash
# Debug: Qwen2.5-1.5B, 1 node, 16k, 15 steps — test for step 10 hang
set -euo pipefail
cd /home/mormio/RL

CONTAINER="nvcr.io#nvidia/nemo-rl:v0.5.0"
EXP="results/erdos-debug-16k-$(date +%Y%m%d_%H%M)"
mkdir -p "$EXP"

MOUNTS="$PWD:$PWD,/home/shared/models:/home/shared/models"

COMMAND="
export HF_HUB_ENABLE_HF_TRANSFER=0
export TORCH_CUDA_ARCH_LIST='9.0 10.0'
export NRL_IGNORE_VERSION_MISMATCH=1
export PYTHONPATH=/home/mormio/RL:\${PYTHONPATH:-}
export ERDOS_LOG_DIR=/home/mormio/RL/results/erdos_debug_outputs

SRC=/home/mormio/RL
cp \$SRC/nemo_rl/algorithms/entropic_advantage_estimator.py /opt/nemo-rl/nemo_rl/algorithms/
cp \$SRC/nemo_rl/environments/erdos_discovery_environment.py /opt/nemo-rl/nemo_rl/environments/
cp \$SRC/nemo_rl/utils/puct_buffer.py /opt/nemo-rl/nemo_rl/utils/
cp \$SRC/examples/run_discover.py /opt/nemo-rl/examples/
cp \$SRC/examples/configs/grpo_erdos_debug_16k.yaml /opt/nemo-rl/examples/configs/

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

cd /opt/nemo-rl
python examples/run_discover.py \
  --config examples/configs/grpo_erdos_debug_16k.yaml
"

echo "Launching debug: Qwen2.5-1.5B, 1 node, 16k, 15 steps"

COMMAND="$COMMAND" \
CONTAINER="$CONTAINER" \
MOUNTS="$MOUNTS" \
GPUS_PER_NODE=8 \
sbatch \
  --nodes=1 --partition=batch --exclusive \
  --job-name=erdos-debug-16k --time=02:00:00 \
  --output="$EXP/slurm-%j.out" \
  --error="$EXP/slurm-%j.err" \
  --exclude=d2dfac12-001,d2dfac12-002,d2dfac12-004,d2dfac12-007,d2dfac12-008,d2dfac12-019,d2dfac12-027,d2dfac12-028,d2dfac12-029 \
  ray.sub

echo "Logs: $EXP/"
