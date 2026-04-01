#!/bin/bash
set -euo pipefail
cd /home/mormio/RL

CONTAINER="/home/shared/containers/nemo-rl-super-v3.sqsh"
MODEL="/home/shared/models/gpt-oss-120b-bf16"
MOUNTS="$PWD:$PWD,/home/shared/models:/home/shared/models"

# Use uv run which activates the right venv with vLLM
COMMAND="
cd /opt/nemo-rl
uv run python -c \"
from vllm import LLM
print('Attempting to load gpt-oss-120b...')
try:
    llm = LLM(
        model='$MODEL',
        tensor_parallel_size=8,
        trust_remote_code=True,
        max_model_len=1024,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )
    print('SUCCESS: gpt-oss-120b loaded!')
    out = llm.generate(['Hello world'], max_tokens=10)
    print('Generated:', out[0].outputs[0].text)
except Exception as e:
    print(f'FAILED: {type(e).__name__}: {e}')
\"
"

COMMAND="$COMMAND" \
CONTAINER="$CONTAINER" \
MOUNTS="$MOUNTS" \
GPUS_PER_NODE=8 \
sbatch \
  --nodes=1 --partition=batch --exclusive \
  --job-name=test-gptoss --time=00:30:00 \
  --output=logs/test-gptoss-%j.out \
  --error=logs/test-gptoss-%j.err \
  --exclude=d2dfac12-001,d2dfac12-002,d2dfac12-004,d2dfac12-007,d2dfac12-008,d2dfac12-019,d2dfac12-027,d2dfac12-028,d2dfac12-029 \
  ray.sub
