#!/bin/bash
# TTT-Discover Erdős GRPO Debug — FINAL
# Only adds our new files to the container's /opt/nemo-rl, does NOT overwrite existing code
set -euo pipefail
cd /home/mormio/RL

CONTAINER="nvcr.io#nvidia/nemo-rl:v0.5.0"
EXP="results/erdos-debug-$(date +%Y%m%d_%H%M)"
mkdir -p "$EXP"

MOUNTS="$PWD:/home/mormio/RL,/home/shared/models:/home/shared/models"

# Only add NEW files to the container's /opt/nemo-rl
# Do NOT overwrite existing files (they match the container's deps)
COMMAND='
export HF_HUB_ENABLE_HF_TRANSFER=0
export TORCH_CUDA_ARCH_LIST="9.0 10.0"
export NRL_IGNORE_VERSION_MISMATCH=1

SRC=/home/mormio/RL

# Add only our NEW files (not overwriting anything)
cp $SRC/nemo_rl/algorithms/entropic_advantage_estimator.py /opt/nemo-rl/nemo_rl/algorithms/
cp $SRC/nemo_rl/environments/erdos_discovery_environment.py /opt/nemo-rl/nemo_rl/environments/
cp $SRC/nemo_rl/utils/puct_buffer.py /opt/nemo-rl/nemo_rl/utils/
cp $SRC/examples/run_discover.py /opt/nemo-rl/examples/
cp $SRC/examples/configs/grpo_erdos_discover_debug.yaml /opt/nemo-rl/examples/configs/

# Patch the container grpo.py to register our entropic estimator
# (append the elif branch to _create_advantage_estimator)
python -c "
path = \"/opt/nemo-rl/nemo_rl/algorithms/grpo.py\"
with open(path) as f:
    content = f.read()
if \"entropic_adaptive_beta\" not in content:
    old = \"    else:\\n        raise ValueError(f\\\"Invalid adv_estimator name: {adv_estimator_name}\\\")\\n\\n    return adv_estimator\"
    new = \"\"\"    elif adv_estimator_name == \\\"entropic_adaptive_beta\\\":
        from nemo_rl.algorithms.entropic_advantage_estimator import (
            EntropicAdaptiveBetaAdvantageEstimator,
        )
        adv_estimator = EntropicAdaptiveBetaAdvantageEstimator(
            adv_estimator_config, loss_config
        )
        print(\\\"  Using Entropic Adaptive-Beta advantage estimator (TTT-Discover)\\\")
    else:
        raise ValueError(f\\\"Invalid adv_estimator name: {adv_estimator_name}\\\")

    return adv_estimator\"\"\"
    content = content.replace(old, new)
    with open(path, \"w\") as f:
        f.write(content)
    print(\"Patched grpo.py with entropic_adaptive_beta\")
else:
    print(\"grpo.py already patched\")
"

# Patch environments/utils.py to register erdos_discovery
python -c "
path = \"/opt/nemo-rl/nemo_rl/environments/utils.py\"
with open(path) as f:
    content = f.read()
if \"erdos_discovery\" not in content:
    content = content.replace(
        \"\\\"nemo_gym\\\": {\",
        \"\\\"erdos_discovery\\\": {\\n        \\\"actor_class_fqn\\\": \\\"nemo_rl.environments.erdos_discovery_environment.ErdosDiscoveryEnvironment\\\",\\n    },\\n    \\\"nemo_gym\\\": {\"
    )
    with open(path, \"w\") as f:
        f.write(content)
    print(\"Patched utils.py with erdos_discovery\")
else:
    print(\"utils.py already patched\")
"

cd /opt/nemo-rl
python examples/run_discover.py \
  --config examples/configs/grpo_erdos_discover_debug.yaml
'

echo "Submitting Erdős TTT-Discover debug..."
echo "Experiment dir: $EXP"

COMMAND="$COMMAND" \
CONTAINER="$CONTAINER" \
MOUNTS="$MOUNTS" \
GPUS_PER_NODE=8 \
sbatch \
  --nodes=2 --partition=batch --exclusive \
  --job-name=erdos-debug --time=01:00:00 \
  --output="$EXP/slurm-%j.out" \
  --error="$EXP/slurm-%j.err" \
  --exclude=d2dfac12-001,d2dfac12-002,d2dfac12-004,d2dfac12-007,d2dfac12-008,d2dfac12-019,d2dfac12-027,d2dfac12-028,d2dfac12-029 \
  ray.sub

echo "Logs: $EXP/"
