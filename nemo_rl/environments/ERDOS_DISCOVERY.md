# TTT-Discover: Learning to Discover at Test Time

Implementation of [TTT-Discover](https://arxiv.org/abs/2601.16175) using NeMo RL
for GRPO training and NeMo Gym for environment scoring.

## Overview

TTT-Discover is a framework for using LLMs to make mathematical discoveries
through iterative refinement. The model generates candidate solutions (as code),
which are scored and organized in a tree structure (PUCT buffer). Training uses
entropic adaptive-β advantages instead of standard GRPO group-relative baselines.

The first application is the **Erdős Minimum Overlap Problem**: finding step
functions that minimize the upper bound on the Erdős constant `c` (known bounds:
`0.379005 < c < 0.380927`).

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  NeMo RL (training)              │
│                                                  │
│  GRPO Loop:                                      │
│    1. Dataloader → prompts from PUCT buffer      │
│    2. vLLM generates code completions            │
│    3. Environment returns rewards                │
│    4. Entropic adaptive-β advantage estimator    │
│    5. Policy gradient update                     │
│    6. Weight sync to vLLM                        │
│                                                  │
│  Components in nemo_rl/:                         │
│    algorithms/entropic_advantage_estimator.py    │
│    environments/erdos_discovery_environment.py   │
│    utils/puct_buffer.py                          │
└─────────────┬───────────────────────────────────┘
              │ HTTP (POST /verify, /select_state, ...)
              │
┌─────────────▼───────────────────────────────────┐
│              NeMo Gym (environment)              │
│                                                  │
│  Erdős Resource Server:                          │
│    - Sandboxed Python code execution             │
│    - FFT-based bound computation                 │
│    - Constraint validation (len, range, mean)    │
│    - reward = 1 / bound                          │
│    - PUCT buffer state management                │
│    - Prompt formatting from tree context         │
│                                                  │
│  Location: Gym/resources_servers/erdos_discovery │
└─────────────────────────────────────────────────┘
```

## Key Components

### Entropic Adaptive-β Advantage Estimator

Location: `nemo_rl/algorithms/entropic_advantage_estimator.py`

Replaces standard GRPO group-relative advantages with Leave-One-Out (LOO)
entropic weighting:

1. **Solve for β**: Find β such that `KL(softmax_β(R) || uniform) = ln(2)`
   via bisection search.
2. **LOO advantages**: `w_i = exp(β·r_i) / Z_{-i} - 1` where `Z_{-i}` is the
   leave-one-out normalizer (excludes sample `i`).

Properties: shift-invariant, approximately scale-invariant, monotone, ~zero-mean.

Config:
```yaml
grpo:
  adv_estimator:
    name: entropic_adaptive_beta
    gamma: 0.6931  # ln(2), target KL divergence
```

### PUCT Buffer

Location: `nemo_rl/utils/puct_buffer.py`

Tree-structured state selection using Predictor + Upper Confidence bounds for
Trees (PUCT). Balances exploitation (high-reward states) with exploration
(under-visited branches).

Score: `Q(s) + c · P(s) · √(1+T) / (1+n(s))`

- `Q(s)`: best reward reachable from state `s`
- `P(s)`: rank-based prior
- `n(s)`: visit count
- `T`: total visits
- `c`: exploration constant (default 1.0)

This is a **general utility** — usable by any iterative optimization environment,
not just Erdős.

### Erdős Discovery Environment

Location: `nemo_rl/environments/erdos_discovery_environment.py`

Ray remote actor implementing `EnvironmentInterface`. Calls the NeMo Gym Erdős
resource server for sandboxed code execution and reward computation.

Config:
```yaml
env:
  erdos_discovery:
    resource_server_url: http://localhost:8080
    num_initial_states: 16
    sandbox_timeout: 600
```

### Erdős Gym Resource Server

Location: `Gym/resources_servers/erdos_discovery/`

Standalone FastAPI server handling:
- `/verify`: execute code, validate `f`, compute `reward = 1/bound`
- `/seed_session`: initialize PUCT buffer with random states
- `/select_state`: PUCT-select states for next training batch
- `/update_buffer`: add discoveries to the tree

## Hyperparameters (from the paper)

| Parameter | Value | Description |
|-----------|-------|-------------|
| model | 120B MoE | gpt-oss-120b-bf16 with LoRA r=32 |
| group_size | 64 | Rollouts per initial state |
| groups_per_batch | 8 | PUCT-selected states per step |
| epochs | 50 | Training steps |
| lr | 4e-5 | Learning rate |
| context_window | 32768 | Max tokens |
| kl_penalty | 0.1 | KL penalty coefficient |
| puct_c | 1.0 | PUCT exploration constant |
| entropic γ | ln(2) | Target KL for adaptive β |
| sandbox_timeout | 600s | Code execution limit |

## Running

_Run script coming soon. Will follow the `research/template_project/` pattern._

```bash
# 1. Start the Gym resource server
cd ~/Gym
ng_run "+config_paths=[resources_servers/erdos_discovery/configs/erdos_discovery.yaml]"

# 2. Run GRPO training with TTT-Discover
cd ~/RL
python research/ttt_discover/run_discover.py \
  --config research/ttt_discover/configs/erdos_120b.yaml
```

## References

- Yu Sun et al., "Learning to Discover at Test Time" (arXiv:2601.16175), 2026.
- Reference implementation: https://github.com/test-time-training/discover
- Haugland (2016) for prior SOTA bound 0.380927.
