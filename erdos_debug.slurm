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

# TTT-Discover debug run — single node, Qwen3-1.7B, 5 training steps.
# Tests the full pipeline: PUCT → prompts → vLLM → sandbox → reward → entropic advantages → gradient

set -euo pipefail
mkdir -p logs

echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi -L | wc -l)"

# ── Gym resource server (background) ─────────────────────────────
# Start the Erdős Gym server on this node (CPU-only, no GPU needed).
# It handles code execution, reward computation, and PUCT state management.
cd /home/mormio/Gym

echo "Starting Gym Erdős resource server on port 8080..."
# TODO: adjust this to your Gym environment setup
# For now, run the FastAPI server directly:
PYTHONPATH=/home/mormio/Gym:$PYTHONPATH \
nohup python -m uvicorn \
    resources_servers.erdos_discovery.app:app \
    --host 0.0.0.0 --port 8080 \
    > logs/erdos-gym-${SLURM_JOB_ID}.log 2>&1 &
GYM_PID=$!
echo "Gym server PID: $GYM_PID"

# Wait for Gym server to be ready
for i in $(seq 1 30); do
    if curl -s http://localhost:8080/docs > /dev/null 2>&1; then
        echo "Gym server ready"
        break
    fi
    sleep 2
done

# ── NeMo RL training ─────────────────────────────────────────────
cd /home/mormio/RL

echo "Starting TTT-Discover GRPO debug training..."
uv run python examples/run_discover.py \
    examples/configs/grpo_erdos_discover_debug.yaml

echo "Training complete"

# Cleanup
kill $GYM_PID 2>/dev/null || true
