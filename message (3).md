# NeMo RL GRPO on Our Cluster — What It Actually Took

## The Ask
Run the NVIDIA NeMo RL GRPO tutorial on our B200 cluster.

## The Cluster
- 32 nodes, 8x B200 (183GB) per node
- Slurm + Pyxis/Enroot for containers
- Shared home directory (NFS), no /scratch
- InfiniBand networking (mlx5 HCAs, bond0)

## What Worked Immediately
- Cloning the repo, downloading models from HuggingFace
- The `ray.sub` script for orchestrating Ray clusters via Slurm (with patches)

## What Needed Fixing

### Cluster-Specific Patches to ray.sub
Every run needed these two fixes to `ray.sub`:
```bash
# 1. MPI plugin: cluster has pmi2, not pmix
sed -i 's/--mpi=pmix/--mpi=pmi2/' ray.sub

# 2. Container filesystem must be writable (ray writes launch scripts to /)
sed -i '/--no-container-mount-home/a COMMON_SRUN_ARGS+=" --container-writable"' ray.sub
```

Also needed to remove `--no-container-mount-home` so the shared home dir 
(with model conversion cache) is visible across all nodes.

### NGC Container Authentication
Pyxis/enroot needs NGC credentials to pull containers:
```bash
# ~/.config/enroot/.credentials
machine nvcr.io login $oauthtoken password <NGC_API_KEY>
```
The image URI format for Pyxis is `nvcr.io#nvidia/nemo-rl:v0.5.0` (note the `#`).

### NCCL / InfiniBand Configuration
Multi-node training was getting `Network is unreachable` NCCL errors until
we added the cluster's IB config (copied from our existing torchtitan slurm scripts):
```bash
export NCCL_IB_HCA=mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_13:1,mlx5_14:1,mlx5_15:1
export NCCL_SOCKET_IFNAME=bond0
export UCX_NET_DEVICES=bond0
export NCCL_BUFFSIZE=33554432
export NCCL_IB_AR_THRESHOLD=0
export NCCL_IB_PCI_RELAXED_ORDERING=1
# ... etc
```
These must be set in `ray.sub` (not just the COMMAND) so every node gets them.

### HF Transfer
The v0.5.0 container sets `HF_HUB_ENABLE_HF_TRANSFER=1` but the package
isn't installed. Set `HF_HUB_ENABLE_HF_TRANSFER=0` in the launch command.

## The Llama 8B Math Run (worked quickly)
**Container**: `nvcr.io/nvidia/nemo-rl:v0.5.0`  
**Branch**: `v0.5.0` tag  
**Config**: `examples/configs/grpo_math_8B_megatron.yaml`  
**Script**: `examples/run_grpo_math.py`  

This used OpenMathInstruct-2 (auto-downloads), single node, colocated generation.
Worked after fixing the ray.sub patches above and enabling W&B
(`++logger.wandb_enabled=true`). The base config has `wandb_enabled: false`.

## The Workplace Assistant Tutorial (abandoned)
The original tutorial targets Nemotron Nano 9B v2 with the Workplace Assistant
NeMo Gym environment. This was a nightmare:

- **v0.5.0 container + v0.5.0 code**: Chat template tokenization assertion errors
  (`non-monotonically increasing trajectory`). The Nemotron Nano v2 tokenizer
  handles multi-turn tool-calling conversations in a way that breaks the
  `_replace_prefix_tokens` function during multi-step rollouts.
- **nano-v3 branch + v0.4.0.nemotron_3_nano container**: The `nemotron_json`
  tool parser wasn't registered in the container's vLLM.
- The tutorial's `sed` commands to patch the chat template are insufficient.
  The real fix exists only on the `nano-v3` branch which removes the assertion entirely.

**Lesson**: The Workplace Assistant environment is tightly coupled to specific
branch/container combos. Use any other environment instead.

## The Nemotron 3 Super 120B Run (what finally worked)

### Container Build
The `super-v3` branch requires a custom container build because it uses a
patched vLLM for the NemotronH MoE architecture:

```bash
# On a compute node (docker access required):
docker buildx build \
  --build-context nemo-rl=. \
  --build-arg SKIP_SGLANG_BUILD=1 \
  --build-arg BUILD_CUSTOM_VLLM=1 \
  -f docker/Dockerfile \
  --tag nemo-rl-super:v3 --load .

# Convert to sqsh for Pyxis:
sudo enroot import -o nemo-rl-super-v3.sqsh "dockerd://nemo-rl-super:v3"
```

We had to install `docker-buildx` first (not available on the cluster by default).

### HF→Megatron Model Conversion
First run converts the HuggingFace checkpoint to Megatron format (~231GB).
This is cached at `~/.cache/huggingface/nemo_rl/model__<sanitized_path>/`.
**The home dir must be mounted in the container** for this cache to be shared
across nodes. Previous runs with `--no-container-mount-home` caused the
conversion to succeed on the head node but be invisible to training nodes.

### Chat Template
The base model (`NVIDIA-Nemotron-3-Super-120B-A12B-Base-BF16`) has no chat
template. The `math_hf_data_processor` calls `tokenizer.apply_chat_template()`,
which crashes. We added a minimal one:

```python
data["chat_template"] = "{% for message in messages %}..."
```

### Data
The internal NVIDIA data paths in the configs (`/lustre/fsw/...`) don't exist.
We downloaded DAPO-Math-17k from HuggingFace and converted it, but ultimately
used the built-in `OpenMathInstruct-2` dataset with `math_hf_data_processor`
and `env.math` (rule-based math verification, no LLM judge needed).

The base config also sets `data.max_input_seq_length: null` which causes a
`TypeError: '>' not supported between instances of 'int' and 'NoneType'`.
Override with `++data.max_input_seq_length=4096`.

### NeMo Gym
The base `grpo_superv3.yaml` config has `env.nemo_gym.num_gpu_nodes: 4` which
reserves 4 GPU nodes for Gym environment servers (genrm judges, etc.). With
only 6 total nodes this left negative nodes for training. Either:
- Set `++env.nemo_gym.num_gpu_nodes=0` if using simple environments
- Set `++env.should_use_nemo_gym=false` to skip Gym entirely

The container's built-in `nemo_gym` package also had an `ImportError` 
(`cannot import name 'PARENT_DIR'`) when the Gym submodule from our repo
checkout was mounted over the container's version. Don't mount the Gym dir.

### Parallelism (the hard part)
The 120B MoE model needs careful parallelism to fit in memory and divide evenly:

**What didn't work:**
- TP=4, CP=4, PP=2: PP=2 from base config made TP×CP×PP=32 > 16 training GPUs
- TP=4, CP=4, PP=1, EP=8, 13 nodes: `World size (40) not divisible by 32`
- TP=4, CP=1, PP=1, EP=2: OOM (105GB allocation, only 32GB free per GPU)
- TP=4, CP=4, PP=1, EP=8, 4 training nodes: Worked for generation+logprobs, 
  but CUDA illegal memory access during backward (cross-node NCCL before IB fix)
- TP=4, CP=1, PP=1, EP=8 (after IB fix): OOM during training backward pass

**What worked:**
```yaml
# 6 nodes: 2 inference, 4 training (32 GPUs)
tensor_model_parallel_size: 4   # within-node
pipeline_model_parallel_size: 1
context_parallel_size: 1        # no cross-node context parallel
expert_model_parallel_size: 8   # MoE experts sharded across all 32 GPUs
# TP×PP×CP×EP = 4×1×1×8 = 32 = world_size, DP=1

# Memory optimizations required:
activation_checkpointing: true
empty_unused_memory_level: 2
optimizer_cpu_offload: true
max_total_sequence_length: 4096  # reduced from 16384
train_micro_batch_size: 1
logprob_batch_size: 1
num_prompts_per_step: 16         # reduced from 128
num_generations_per_prompt: 8    # reduced from 16
train_global_batch_size: 128     # reduced from 2048
```

### Final Working Command
```bash
cd /opt/nemo-rl && \
export NCCL_IB_HCA=mlx5_4:1,mlx5_7:1,... && \
export NCCL_SOCKET_IFNAME=bond0 && \
uv run python examples/run_grpo.py \
  --config=examples/configs/grpo_superv3.yaml \
  ++env.should_use_nemo_gym=false \
  ++data.train.dataset_name=OpenMathInstruct-2 \
  ++data.default.processor=math_hf_data_processor \
  ++data.default.env_name=math \
  ++env.math.num_workers=8 \
  ++env.math.math_verify_impl=hf_math_verify \
  ++policy.model_name=/path/to/model \
  ++cluster.num_nodes=6 \
  ++policy.generation.colocated.enabled=false \
  ++policy.generation.colocated.resources.num_nodes=2 \
  ++policy.megatron_cfg.tensor_model_parallel_size=4 \
  ++policy.megatron_cfg.pipeline_model_parallel_size=1 \
  ++policy.megatron_cfg.context_parallel_size=1 \
  ++policy.megatron_cfg.expert_model_parallel_size=8 \
  ++policy.megatron_cfg.activation_checkpointing=true \
  ++policy.megatron_cfg.optimizer.optimizer_cpu_offload=true \
  ++policy.max_total_sequence_length=4096 \
  # ... etc
```

~120 seconds per step on 6x B200 nodes (48 GPUs).

## TL;DR
1. Fix `ray.sub` for your cluster (MPI plugin, container writable, home mount)
2. Set NCCL IB env vars for multi-node
3. Don't use the Workplace Assistant tutorial — use math environments instead
4. For Nemotron Super: build container from `super-v3` branch, use EP=8 to shard MoE experts, offload optimizer to CPU, reduce batch sizes
5. The model conversion cache needs to be on a shared filesystem visible to all nodes