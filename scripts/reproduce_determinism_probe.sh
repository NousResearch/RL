#!/bin/bash
# Reproduce the same-program determinism probe: Multi twice + true single-A
# twice, byte-identical except run_name, on one node.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
CFG_DIR="${REPO_ROOT}/examples/configs/recipes/multi_lora"
LAUNCHER="${CFG_DIR}/sft_8gpu_native.slurm"
CANON="${CANON:-/home/phuc/workspace/rl/small_prs/pr001_tinker_mvp/nousnet_pre_multilora_8178f9c/results/code7x_exactinit_canonical}"
NODE="${NODE:?Set NODE to one healthy idle 8-GPU node, e.g. NODE=d2dfac12-005}"
PREFIX="${PREFIX:-detprobe_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ -d "${CANON}" ]] || { echo "ERROR: canonical init missing: ${CANON}" >&2; exit 1; }
BASE_ENV="NOUSNET_DIAG_ENABLED=1,NOUSNET_DIAG_LOSS_TRACE=1,NOUSNET_DIAG_TRACE_ONLY=1,NOUSNET_DIAG_LORA_STEP=0,NOUSNET_INIT_IMPORT_DIR=${CANON}"
submit() {
  local who="$1" cfg="$2" slot="$3" diag_who="$4" jid
  jid=$(sbatch --parsable -w "${NODE}" \
    --export=ALL,CFG="${cfg}",${BASE_ENV},NOUSNET_INIT_IMPORT_SLOT="${slot}",NOUSNET_DIAG_WHO="${diag_who}",NOUSNET_PER_ADAPTER_GRAD_CLIP=0 \
    -J "${PREFIX}-${who}" "${LAUNCHER}")
  printf '%s=%s ' "${who}" "${jid}"
}
printf '%s ' "$(submit m1 "${CFG_DIR}/detprobe_multi_1.yaml" '' multi)"
printf '%s ' "$(submit m2 "${CFG_DIR}/detprobe_multi_2.yaml" '' multi)"
printf '%s ' "$(submit a1 "${CFG_DIR}/detprobe_single_a_1.yaml" 0 single_a)"
printf '%s ' "$(submit a2 "${CFG_DIR}/detprobe_single_a_2.yaml" 0 single_a)"
printf '\n'
