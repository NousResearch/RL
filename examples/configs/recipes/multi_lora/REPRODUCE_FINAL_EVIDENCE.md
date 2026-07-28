# Reproduce the final repo-native Multi-LoRA evidence

This bundle reproduces the closing evidence documented in `PARITY_REPORT.md`:

1. 100-step cross-node parity: packed Multi-LoRA + true singles A/B/C/D.
2. 10-step same-node parity control: the same five lanes serialized on one node.
3. Same-program determinism probe: Multi twice + true single-A twice on one node.

All comparisons are pairwise A↔A/B↔B/C↔C/D↔D. Adapter losses are never summed.

## Prerequisites

```bash
REPO=/home/phuc/workspace/rl/small_prs/pr001_tinker_mvp/RL_multilora_native
CONTAINER=/home/shared/containers/nemo-rl-super-v3.sqsh
CANON=/home/phuc/workspace/rl/small_prs/pr001_tinker_mvp/nousnet_pre_multilora_8178f9c/results/code7x_exactinit_canonical
DATA=/home/phuc/multi_lora_data

cd "$REPO"
git checkout c0841cbe35641f4fb9e41d5d4f557844c6e54731

test -f "$CONTAINER"
test -d "$CANON"
for f in "$DATA"/hellaswag_train_{a,b,c,d}.jsonl "$DATA"/hellaswag_val.jsonl; do
  test -f "$f"
done
```

Expected dataset SHA-256:

```text
0bd6c2dd57feb24f1e4a8108241c63e1ae3d7eed8a07c1064956442ce003a0b3  hellaswag_train_a.jsonl
503ead5e00210c580cca07814db510a65d2e8b7c08c09beb014fd20ec06de386  hellaswag_train_b.jsonl
272f0adefbc114437f1202720b802ed2c47f146322b0ab26e39253c17edfd5d5  hellaswag_train_c.jsonl
e079d63a0e6aaa072a945879de93f9feadd649b7bd93ff127a84ea5922493400  hellaswag_train_d.jsonl
53943c4dec82f0c95240312ccaabdcb09a2f9f7079cf35e77bd77adab36e717e  hellaswag_val.jsonl
```

## 1. Reproduce the 100-step battery

```bash
cd /home/phuc/workspace/rl/small_prs/pr001_tinker_mvp/RL_multilora_native
CANON=/home/phuc/workspace/rl/small_prs/pr001_tinker_mvp/nousnet_pre_multilora_8178f9c/results/code7x_exactinit_canonical \
  bash scripts/reproduce_final_100step_parity.sh
```

The command prints five job IDs. When complete, identify each run directory by its job name and ID under `results/`, then analyze:

```bash
python3 scripts/analyze_pairwise_loss.py \
  /absolute/path/to/multi-run \
  /absolute/path/to/single-a-run \
  /absolute/path/to/single-b-run \
  /absolute/path/to/single-c-run \
  /absolute/path/to/single-d-run \
  --steps 100 --dp-size 8 \
  --output-prefix results/repro100_pairwise \
  --title "Repo-native 100-step no-clip parity"
```

Reference jobs were `256236–256240`. Reference maxima were A `0.021683734`, B `0.063501569`, C `0.047479308`, D `0.337963845` at step 57. Numerical reruns are not expected to be bit-identical; see probe 3.

## 2. Reproduce the same-node 10-step control

Select one healthy idle 8-GPU node and pin all lanes to it:

```bash
cd /home/phuc/workspace/rl/small_prs/pr001_tinker_mvp/RL_multilora_native
NODE=d2dfac12-018 \
CANON=/home/phuc/workspace/rl/small_prs/pr001_tinker_mvp/nousnet_pre_multilora_8178f9c/results/code7x_exactinit_canonical \
  bash scripts/reproduce_final_10step_samenode.sh
```

The five jobs serialize because each requests all eight GPUs on the same node. Analyze:

```bash
python3 scripts/analyze_pairwise_loss.py \
  /absolute/path/to/multi-run \
  /absolute/path/to/single-a-run \
  /absolute/path/to/single-b-run \
  /absolute/path/to/single-c-run \
  /absolute/path/to/single-d-run \
  --steps 10 --dp-size 8 \
  --output-prefix results/repro10_samenode_pairwise \
  --title "Repo-native same-node 10-step parity"
```

Reference jobs were `256221–256225`, all on `d2dfac12-018`; A/B/C/D all passed `<0.1`.

## 3. Reproduce the same-program determinism probe

```bash
cd /home/phuc/workspace/rl/small_prs/pr001_tinker_mvp/RL_multilora_native
NODE=d2dfac12-005 \
CANON=/home/phuc/workspace/rl/small_prs/pr001_tinker_mvp/nousnet_pre_multilora_8178f9c/results/code7x_exactinit_canonical \
  bash scripts/reproduce_determinism_probe.sh
```

This launches `M1`, `M2`, `A1`, and `A2` on one node. Compare trace rows by `(step,input_sha256)`:

- `A1↔A2` measures ordinary single-LoRA reproducibility.
- `M1↔M2` separately measures each packed adapter's reproducibility.
- `M1 slot A↔A1` is the Multi-versus-single gate.

Reference jobs were `256326–256329`. The reference showed run-to-run nondeterminism in both paths: only 6/8 single and 5/8 Multi step-1 ranks were bit-exact.

## Execution-success verification

Do not classify from `sacct` alone. Ray teardown can leave `FAILED 15:0` after successful training. For every lane require:

```bash
RUN=/absolute/path/to/results/job-name-job-id
LOG=$(find "$RUN" -path '*-logs/ray-head.log' -print -quit)
LOGDIR=$(dirname "$LOG")

test "$(cat "$LOGDIR/COMMAND_EXIT_CODE")" = 0
test -f "$LOGDIR/COMMAND_DONE"
grep -n '^SFT_DONE_OK$' "$LOG"
grep -n 'SFT training complete\.' "$LOG"
```

Also verify exact trace cardinality:

- 100-step Multi: 3,200 rows; each single: 800 rows.
- 10-step Multi: 320 rows; each single: 80 rows.
- rank set exactly `0..7`.
- final Multi logs must contain `MULTILORA_FSDP_PLACEMENT_OK` for ranks `0..7` with one source SHA.

## Focused tests

```bash
cd /home/phuc/workspace/rl/small_prs/pr001_tinker_mvp/RL_multilora_native
sbatch results/final_delivery/run_unit_tests.slurm
```

Reference result: `200 passed`; full `tests/unit` collected 1,853 tests but was not executed.
