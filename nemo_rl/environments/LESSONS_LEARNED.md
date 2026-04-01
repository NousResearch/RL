# TTT-Discover on NeMo RL — Lessons Learned

## What This Is

TTT-Discover (arXiv:2601.16175) running the Erdős Minimum Overlap Problem
on NeMo RL's GRPO framework. The LLM writes Python code defining a step
function, which is executed in a sandbox and scored via FFT autocorrelation.

## What Worked

### Final Working Setup
- **Container**: `nvcr.io#nvidia/nemo-rl:v0.5.0`
- **Cluster**: 2 nodes, 8x B200 per node (16 GPUs total)
- **Model**: Qwen/Qwen2.5-1.5B-Instruct with LoRA (r=16) for debug
- **Ray orchestration**: Dakota's patched `ray.sub` (see below)
- **~27s per training step** (5 debug steps completed successfully)

### Launch Pattern
```bash
cd ~/RL

CONTAINER="nvcr.io#nvidia/nemo-rl:v0.5.0"
MOUNTS="$PWD:$PWD,/home/shared/models:/home/shared/models"
COMMAND="
export HF_HUB_ENABLE_HF_TRANSFER=0
export TORCH_CUDA_ARCH_LIST='9.0 10.0'
export PYTHONPATH=/path/to/RL:\${PYTHONPATH:-}
cd /opt/nemo-rl
python examples/run_discover.py --config examples/configs/grpo_erdos_discover_debug.yaml
"

COMMAND="$COMMAND" CONTAINER="$CONTAINER" MOUNTS="$MOUNTS" GPUS_PER_NODE=8 \
sbatch --nodes=2 --partition=batch --exclusive --time=01:00:00 ray.sub
```

### Key Config (grpo_erdos_discover_debug.yaml)
```yaml
defaults: "grpo_math_1B.yaml"   # Inherit all base settings

grpo:
  num_prompts_per_step: 4       # 4 PUCT-selected states
  num_generations_per_prompt: 8 # 8 rollouts per state
  max_num_steps: 5              # Debug: 5 steps only
  max_rollout_turns: 1          # Single-turn code generation
  adv_estimator:
    name: entropic_adaptive_beta  # NOT standard grpo
    gamma: 0.6931471805599453     # ln(2)

policy:
  model_name: "Qwen/Qwen2.5-1.5B-Instruct"
  max_total_sequence_length: 4096
  dtensor_cfg:
    cpu_offload: false    # IMPORTANT: must be false or logprobs crash
  lora_cfg:
    enabled: true
    rank: 16

env:
  erdos_discovery:
    resource_server_url: "inline"  # No Gym server needed for debug
```

## Cluster-Specific Fixes (d2dfac12 / Together AI B200)

### ray.sub Patches
Use Dakota's patched `ray.sub` from `~/dakota-ref/ray.sub`. Key changes from upstream:
1. **MPI**: `--mpi=pmi2` (not `pmix`)
2. **Container writable**: `--container-writable` instead of `--no-container-mount-home`
3. **NCCL IB config** in ray.sub itself (not just COMMAND):
   ```bash
   export NCCL_IB_HCA=mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_13:1,mlx5_14:1,mlx5_15:1
   export NCCL_SOCKET_IFNAME=bond0
   export UCX_NET_DEVICES=bond0
   ```
4. **No `--account`** needed on this cluster
5. `srun --overlap` works via `enroot exec` (Dakota's ray.sub handles this)

### NGC Container Auth
```bash
mkdir -p ~/.config/enroot
cat > ~/.config/enroot/.credentials << 'EOF'
machine nvcr.io login $oauthtoken password nvapi-YOUR_KEY_HERE
EOF
```

### Container Pull Time
First pull takes ~10-12 minutes per node. Cached after that, but cache is
per-node (different nodes need their own pull). Plan for this in your first run.

## What Broke and How We Fixed It

### 1. TransformerEngine Won't Build (bare metal)
**Problem**: `uv sync --extra automodel` fails because TransformerEngine needs
`cudnn.h` which doesn't exist on the head node.
**Fix**: Use the NeMo RL container (has TE pre-built). Don't try bare metal
with LoRA — LoRA requires DTensorPolicyWorkerV2 which needs the `automodel`
extra which needs TransformerEngine.

### 2. Ray Version Mismatch
**Problem**: Container has Ray 2.49.2, but `uv run` picks up Ray 2.54.0 from
the mounted `.venv`.
**Fix**: Use `python` directly (container's Python), not `uv run`. Set your
code path via `PYTHONPATH` instead.

### 3. Code Compatibility with v0.5.0 Container
**Problem**: Our branch's `nemo_rl` code is newer than v0.5.0 (has `decord`
imports, `register_omegaconf_resolvers`, etc. that the container doesn't have).
**Fix**: Don't overwrite the container's code at `/opt/nemo-rl`. Instead:
  - Copy only NEW files (custom estimator, environment, run script)
  - Monkey-patch `grpo.py` and `utils.py` at runtime to register new components
  - Add `mul`/`div` OmegaConf resolvers manually if `register_omegaconf_resolvers` isn't available

### 4. Ray Actor Event Loop
**Problem**: `asyncio.get_event_loop().run_until_complete()` in environment
`step()` crashes with "This event loop is already running" inside Ray actors.
**Fix**: Detect running loop and use synchronous path instead:
```python
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = None
if loop and loop.is_running():
    return self._sync_step(...)  # No async
else:
    return asyncio.run(self._async_step(...))
```

### 5. Empty Observations
**Problem**: `KeyError: 'content'` in rollouts.py — the rollout engine expects
observations from `env.step()` to have a `content` key.
**Fix**: Return `[{"role": "user", "content": ""} for _ in range(batch_size)]`
not `[{}] * batch_size`.

### 6. CPU Offload + LogProbs
**Problem**: `RuntimeError: Expected all tensors to be on the same device` —
model weights on CPU (from `cpu_offload: true`) but input_ids on cuda.
**Fix**: Set `dtensor_cfg.cpu_offload: false` for small debug models.
For large models, double-check the offload/reload flow works with your config.

### 7. `${mul:...}` OmegaConf Resolver
**Problem**: Base config uses `${mul:a,b}` interpolation but v0.5.0 container
doesn't register this resolver.
**Fix**: Register it manually in your run script:
```python
from omegaconf import OmegaConf
if not OmegaConf.has_resolver("mul"):
    OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
```

## Architecture

```
NeMo RL Container (v0.5.0)
├── /opt/nemo-rl/           ← Container's code (base)
│   ├── nemo_rl/algorithms/
│   │   ├── grpo.py                          ← monkey-patched at runtime
│   │   └── entropic_advantage_estimator.py  ← our NEW file
│   ├── nemo_rl/environments/
│   │   ├── utils.py                         ← monkey-patched at runtime
│   │   └── erdos_discovery_environment.py   ← our NEW file
│   └── nemo_rl/utils/
│       └── puct_buffer.py                   ← our NEW file
│
├── /home/mormio/RL/        ← Mounted source (for cp at startup)
└── /home/shared/models/    ← Mounted model weights
```

## Scaling to 120B

For gpt-oss-120b-bf16 (MoE, 128 experts):
- Use 8 nodes (2 training + 6 inference) or similar
- LoRA r=32, alpha=1.0
- EP=8 for expert parallelism (see Dakota's 120B config)
- May need `cpu_offload: true` + careful memory management
- Set `max_total_sequence_length: 32768` for full context
- Consider `generation.colocated: false` with separate inference nodes

## References
- Paper: "Learning to Discover at Test Time" (arXiv:2601.16175)
- Reference impl: https://github.com/test-time-training/discover
- Dakota's working configs: `~/dakota-ref/`
- NeMo RL docs: https://docs.nvidia.com/nemo/rl/latest/index.html
