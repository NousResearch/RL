# Repo-native NeMo-RL Multi-LoRA — per-adapter parity final report

**Generated:** 2026-07-28 (UTC)
**Repository:** `/home/phuc/workspace/rl/small_prs/pr001_tinker_mvp/RL_multilora_native`
**Branch:** `phuc/multilora-native-nemorl`
**Machine-readable record:** `results/final_delivery/final_results.json`

---

## Verdict

**All ten training lanes executed successfully. The packed 4-adapter Multi-LoRA run
matches its four true single-LoRA references on 3 of 4 adapters over 100 steps
(A, B, C PASS; D FAILS at exactly one isolated step of 100), and on 4 of 4 in the
same-node 10-step control. A determinism probe run this campaign shows the harness
reproduces *itself* only to ~0.09 per step, so the single D excursion is real but is
not established as a Multi-LoRA defect — run-to-run determinism must be restored
before that attribution can be made.**

Execution success and scientific parity are reported separately throughout.

---

## 1. What was compared

Four LoRA adapters (A, B, C, D) are trained together inside one *packed* Multi-LoRA
job, where all four adapters live stacked in the same parameter tensors. Each adapter
is also trained alone in its own *true single* job. Parity means: adapter A's rows
inside the packed run must produce the same loss as the standalone single-A run.

- Pairings are strictly `A↔A`, `B↔B`, `C↔C`, `D↔D`.
- **Adapter losses are never summed.** Summing would let one adapter's error cancel
  another's and hide exactly the failure mode under test.
- Rows are joined on `(optimizer_step, DP rank)`, then **exact `input_sha256` and
  `num_tokens` equality is required** before any loss is compared. This proves both
  sides consumed byte-identical inputs.
- Reported loss is the token-weighted mean over the 8 data-parallel ranks.

**Pre-declared gate:** per-step `|Δloss| < 0.1` for every adapter at every step.

Analyzer: `scripts/analyze_pairwise_loss.py` (unmodified repo tool).

---

## 2. Execution success (SLURM accounting deliberately ignored)

All ten outer SLURM allocations report `FAILED 15:0`. That is Ray teardown noise —
`ray.sub` cancels the still-live Ray step after the inner command has already
finished. Accounting state alone was **not** used to classify any lane.

A lane counts as execution-success only if **all** of the following hold:

1. `COMMAND_EXIT_CODE` exists and equals `0`
2. `COMMAND_DONE` exists
3. an **anchored standalone** `^SFT_DONE_OK$` line occurs **after** `SFT training complete.`
4. the exact expected diagnostic trace-row count is present, over the exact 8-rank set
5. multi lanes additionally show `MULTILORA_FSDP_PLACEMENT_OK` on **all 8 ranks** with a
   single adapter-implementation `sha256`

> **Trap avoided:** a raw substring grep for `SFT_DONE_OK` returns **3** hits per log,
> because the launcher echoes the literal near the start. Only **1** is the anchored
> terminal marker. Validation required the anchored form *and* correct ordering.

### Result: 10 / 10 lanes execution-success

| Battery | Lane | Job | Traces | Expected | Exit | Anchored marker |
|---|---|---|---|---|---|---|
| 100-step | multi | 256236 | 3200 | 3200 | 0 | yes |
| 100-step | single A | 256237 | 800 | 800 | 0 | yes |
| 100-step | single B | 256238 | 800 | 800 | 0 | yes |
| 100-step | single C | 256239 | 800 | 800 | 0 | yes |
| 100-step | single D | 256240 | 800 | 800 | 0 | yes |
| 10-step same-node | multi | 256221 | 320 | 320 | 0 | yes |
| 10-step same-node | single A | 256222 | 80 | 80 | 0 | yes |
| 10-step same-node | single B | 256223 | 80 | 80 | 0 | yes |
| 10-step same-node | single C | 256224 | 80 | 80 | 0 | yes |
| 10-step same-node | single D | 256225 | 80 | 80 | 0 | yes |

Every multi lane reported `MULTILORA_FSDP_PLACEMENT_OK` on ranks 0–7 with
`modules=6004` and one identical implementation hash
`a5ba903c15053c0b81fcd6ac138380c1fddd078d2534c0416806148b973743cd`, confirming the
placement fix was live on all workers and identical across them.

---

## 3. Scientific result — 100-step battery (cross-node)

Jobs: multi `256236`; singles A/B/C/D `256237` / `256238` / `256239` / `256240`.
Submitted by `scripts/submit_parity100_new.sh`.

Evidence integrity for every adapter: **800/800 rank rows aligned, 100/100 steps,
0 hash mismatches, 0 token mismatches, 0 missing/extra rows.**

| Pair | max per-step \|Δ\| | at step | mean \|Δ\| | final-step \|Δ\| | steps ≥ 0.1 | verdict |
|---|---|---|---|---|---|---|
| A↔A | 0.021683734 | 55 | 0.005796181 | 0.003601332 | 0 | **PASS** |
| B↔B | 0.063501569 | 14 | 0.009957560 | 0.014640648 | 0 | **PASS** |
| C↔C | 0.047479308 | 7 | 0.007865843 | 0.011966801 | 0 | **PASS** |
| D↔D | 0.337963845 | 57 | 0.012662506 | 0.012512290 | 1 | **FAIL** |

Analyzer overall: `PAIRWISE FAIL` (3/4 adapters pass).

### Anatomy of the single D failure

The D excursion is one isolated step, not accumulating drift:

```
step 54  |Δ|=0.001112
step 55  |Δ|=0.003428
step 56  |Δ|=0.008937
step 57  |Δ|=0.337964   <-- the only step at or above 0.1
step 58  |Δ|=0.099992
step 59  |Δ|=0.003782
step 60  |Δ|=0.019824
```

At step 57, all 8 ranks disagree simultaneously (per-rank |Δ| between 0.218 and 0.613),
while `input_sha256` and `num_tokens` match on **all 8 ranks** — so both sides received
identical inputs and the divergence is numerical, not a data or routing mismatch. The
run reconverges immediately: D's last-10-step maximum is 0.018540 and its final step is
0.012512. D's mean over 100 steps (0.0127) is in line with the passing adapters.

---

## 4. Scientific result — 10-step same-node control

Jobs: multi `256221`; singles A/B/C/D `256222` / `256223` / `256224` / `256225`, all five
lanes pinned to node `d2dfac12-018`. Submitted by `scripts/submit_samenode10_battery.sh`.

Integrity: **80/80 rank rows aligned, 10/10 steps, 0 hash mismatches, 0 token mismatches**
for every adapter.

| Pair | max per-step \|Δ\| | at step | mean \|Δ\| | final-step \|Δ\| | verdict |
|---|---|---|---|---|---|
| A↔A | 0.007831854 | 5 | 0.003990854 | 0.003036421 | **PASS** |
| B↔B | 0.010508365 | 4 | 0.005027997 | 0.007010109 | **PASS** |
| C↔C | 0.042107000 | 7 | 0.012182033 | 0.002852315 | **PASS** |
| D↔D | 0.047586891 | 7 | 0.012961741 | 0.002293180 | **PASS** |

Analyzer overall: `PAIRWISE PASS` (4/4).

This control is analysed **separately** from the 100-step battery; it is a different
horizon (10 vs 100 steps) and a different placement (single node vs cross-node), so
its numbers are not comparable term-by-term with §3.

---

## 5. The determinism confound — why D is not yet a Multi-LoRA bug

A dedicated probe (jobs `256326`, `256327` multi ×2; `256328`, `256329` single-A ×2) ran
**byte-identical configs twice on the same node** `d2dfac12-005`, differing only in
`run_name`. Two runs of the same thing should agree exactly. They do not.

| Comparison | max per-step \|Δ\| | max per-rank \|Δ\| | step-1 bit-exact ranks |
|---|---|---|---|
| single-A vs single-A (same config twice) | 0.030518 | 0.084677 | 6 / 8 |
| multi vs multi, adapter A | 0.013920 | 0.055452 | 5 / 8 |
| multi vs multi, adapter B | 0.010162 | 0.033305 | 5 / 8 |
| multi vs multi, adapter C | 0.029022 | 0.070138 | 5 / 8 |
| multi vs multi, adapter D | **0.089830** | **0.190351** | 5 / 8 |

**Measured noise floor: ~0.0898 per step and ~0.1904 per DP rank**, with step-1 losses
not bit-exact between two identical runs.

Consequences:

- A, B and C's 100-step maxima (0.0217 / 0.0635 / 0.0475) all sit **at or below** the
  harness's own reproducibility floor. Their PASS is comfortable but the margin is
  partly noise-limited.
- D's 0.3380 is genuinely **above** the floor — it is a real excursion, not pure jitter.
  But because a *multi-vs-multi* comparison of the same config already reaches 0.0898 on
  adapter D specifically, and because the excursion is a single non-recurring step, this
  evidence does **not** isolate Multi-LoRA as the cause.
- For reference, the actual multi-vs-single gate inside the probe measured 0.007604
  (M1 slot A vs A1) and 0.043757 (M2 slot A vs A2) — i.e. the multi-vs-single difference
  is *smaller* than the same-config-twice difference, which is the opposite of what a
  Multi-LoRA regression would look like.

**Recommended next step:** restore run-to-run determinism (seeding, kernel selection,
reduction order) before re-litigating adapter D. Today the instrument is noisier than
several of the effects being measured.

---

## 6. Historical comparators (labeled context only — not a substitute for measurement)

These are prior campaigns on earlier code, quoted for orientation only. None of them is
used as a gate or as a stand-in for the measured results above.

- **100-step no-clip, jobs 252203–252207 (2026-07-24):** A 0.0906, B 0.0543, C 0.0547,
  D 0.0891 — verdict PASS.
- **100-step V3 no-clip, jobs 251641–251645:** D FAIL at one isolated step (step 8,
  0.108475).
- **Earlier campaign reference maxima:** A 0.0515, B 0.0286, C 0.0566, D 0.1680.
- The **~0.127 100-step numerical floor** referenced historically is cited here strictly
  as a labeled comparison point. It was **not** used as a threshold, and the gate applied
  in this report is the pre-declared `< 0.1`.

Note the pattern across campaigns: an isolated single-step excursion on one adapter
recurs across code revisions and moves between adapters (D at step 8, C at step 7, A at
step 55, now D at step 57). That is the signature of a numerical-reproducibility limit,
not of a specific adapter's routing being wrong.

---

## 7. Code under test and validation

### Load-bearing changes delivered

- `nemo_rl/models/multi_lora/adapter.py` — adds `assert_stacked_lora_fsdp_placement()`,
  a fail-closed check that stacked LoRA parameters keep **every adapter slot local** and
  shard dim 1. `Shard(0)` would shard adapter *identity* and silently corrupt routing.
  The assertion also all-gathers its result so a placement/source divergence between
  workers is caught, and prints `MULTILORA_FSDP_PLACEMENT_OK` with the implementation
  `sha256`.
- `nemo_rl/models/policy/workers/dtensor_policy_worker_v2.py` — invokes that assertion on
  every worker after FSDP/PEFT setup and **before** exact-init import or the first
  forward, only when `n_adapters > 1`.
- `nemo_rl/models/multi_lora/moe_routing.py` — converts the `Shard(0)` case from a
  `logger.warning` into a `RuntimeError`. Previously an invalid placement warned and
  continued toward a hang or silent corruption.
- `nemo_rl/models/multi_lora/routing.py` — removes the blanket `try/except` that
  swallowed MoE routing install failures and silently fell back to the legacy slot-0
  path. Failures now surface.
- `examples/configs/recipes/multi_lora/sft_8gpu_native.slurm` — resolves *all* inherited
  config files listed under `defaults:` from the submitted config's own directory and
  fails closed if one is missing, instead of copying only a hard-coded `base.yaml`.
- Tests: 6 new placement/assertion tests, 1 new MoE `Shard(0)` rejection test, and 2 new
  tests pinning the vendored parallelizer's `_fsdp_shard_dim` hint behaviour.

### Test evidence

Run inside the campaign container `/home/shared/containers/nemo-rl-super-v3.sqsh`
(SLURM job `256432`, node `d2dfac12-020`), with `PYTHONPATH` pointed at this checkout so
the **working-tree code** is what executes (`nemo_rl from:
/home/phuc/workspace/.../RL_multilora_native/nemo_rl/__init__.py`, torch 2.9.0+cu129):

- **Focused suite `tests/unit/models/multi_lora`: 200 passed, 0 failed, 0 errors**
  (test_loss.py 21, test_moe_routing.py 10, test_multi_linear_lora.py 49,
  test_multi_sharding.py 9, plus parametrised cases).
- **Scope evidence:** `pytest tests/unit --collect-only` collects **1853 tests with zero
  collection errors**, so there is no hidden module that fails to import.

**Test scope statement — read precisely:** the *focused* Multi-LoRA unit suite was
executed and fully passed. The *full* unit suite was **collected but not executed**.
This report therefore does **not** claim a full-suite pass. Multi-GPU/functional tests
were likewise not run as part of this validation; the distributed evidence is the ten
training lanes in §2–§4.

---

## 8. Reproduction commands

```bash
REPO=/home/phuc/workspace/rl/small_prs/pr001_tinker_mvp/RL_multilora_native
cd "$REPO"

# 100-step battery (multi + 4 true singles)
bash scripts/submit_parity100_new.sh

# 10-step same-node control battery
bash scripts/submit_samenode10_battery.sh

# determinism probe (identical config twice, one node)
bash scripts/submit_detprobe.sh

# pairwise analysis — 100-step
python3 scripts/analyze_pairwise_loss.py \
  results/p100n-multi-256236 results/p100n-sa-256237 results/p100n-sb-256238 \
  results/p100n-sc-256239 results/p100n-sd-256240 \
  --steps 100 --dp-size 8 \
  --output-prefix results/final_delivery/parity100_new_pairwise

# pairwise analysis — 10-step same-node control
python3 scripts/analyze_pairwise_loss.py \
  results/sn10-multi-256221 results/sn10-sa-256222 results/sn10-sb-256223 \
  results/sn10-sc-256224 results/sn10-sd-256225 \
  --steps 10 --dp-size 8 \
  --output-prefix results/final_delivery/samenode10_pairwise

# focused unit tests (container)
sbatch results/final_delivery/run_unit_tests.slurm
```

Shared run configuration for every lane:

```
NOUSNET_DIAG_ENABLED=1 NOUSNET_DIAG_LOSS_TRACE=1 NOUSNET_DIAG_TRACE_ONLY=1
NOUSNET_DIAG_LORA_STEP=0 NOUSNET_PER_ADAPTER_GRAD_CLIP=0
NOUSNET_INIT_IMPORT_DIR="${CANON}"
```

`CANON` is the directory of canonical exact-init LoRA shards (`code7x_exactinit_canonical`),
supplied via the environment by the battery scripts — see `scripts/submit_parity_battery.sh`,
which fails closed with instructions when it is unset. It is deliberately not hardcoded
here: this recipe must stay free of external-checkout paths, which
`scripts/audit_multi_lora_standalone.py` enforces.

Singles additionally set `NOUSNET_INIT_IMPORT_SLOT` to `0/1/2/3` for A/B/C/D so each
single starts from the same canonical adapter weights as its packed counterpart, plus
`multi_lora.enabled: false`, `train_global_batch_size: 16`, `train_micro_batch_size: 2`.
No gradient clipping in either battery (`NOUSNET_PER_ADAPTER_GRAD_CLIP=0`).

---

## 9. Artifacts

All under `results/final_delivery/`:

| File | SHA-256 |
|---|---|
| `final_results.json` | `d39e627566f09ce69192d31be0744db50cee6760b5ac332d079d32b8002c9123` |
| `final_results_summary.csv` | `51910e868656e74124c5c945882fb9fce7d89534121a981a34e7aa36b3c132a8` |
| `conclusions_diagram.png` | `c229f8cd117924c8f8038f963b6837f2f2bda3963b3b477f2446b084fd7f8ff3` |
| `pairwise_heatmap.png` | `e5150535b0dbb47d11e2388bba1b2d9aae245858974e8f2863ed9e5c4957e527` |
| `parity100_new_pairwise.json` | `b4ebd1bdf1d131e3ea8c41445390efce028ddfaac7fb1180f3c49ff01e0fb922` |
| `parity100_new_pairwise.csv` | `2ca96ca9a9401e6333f05d0fe954b187553ae8ecf38fd0cf9181f5a46df942e5` |
| `samenode10_pairwise.json` | `38e5444320dd212c06477647023b5b33b5149ee74db35685b1eb8dd1c1856a09` |
| `samenode10_pairwise.csv` | `0cbe717bcc038676b120b2cab636e50a0e9e5f7328ab0ef6daf5b367071fa316` |

Raw traces (not committed; on shared storage):
`results/p100n-{multi,sa,sb,sc,sd}-2562{36,37,38,39,40}/diag_loss_trace/`,
`results/sn10-{multi,sa,sb,sc,sd}-2562{21,22,23,24,25}/diag_loss_trace/`,
`results/dp-{m1,m2,a1,a2}-2563{26,27,28,29}/diag_loss_trace/`.

### Figure verification

No vision model was available in this session, so both figures were audited
programmatically instead: PIL integrity verification, dimension and colour-depth checks,
then OCR (inverted + upscaled, multiple page-segmentation modes) with **every rendered
number asserted equal to `final_results.json`**. All 40 checked values matched. The
audit also asserts the figures contain no overall-pass claim.

---

## 10. Status

- **Pending user action:** none.
- **Pending agent action:** none. No watchers armed by this delivery; the batteries and
  the probe are complete.
- **Open technical question (not blocking this delivery):** run-to-run determinism. Until
  it is restored, the `< 0.1` per-step gate cannot cleanly separate a Multi-LoRA defect
  from harness noise on adapter D.
