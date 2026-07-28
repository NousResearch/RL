#!/bin/bash
# Reproduce the final same-node 10-step parity control. All five jobs are pinned
# to one node and therefore serialize there.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
CFG_DIR="${REPO_ROOT}/examples/configs/recipes/multi_lora"
LAUNCHER="${CFG_DIR}/sft_8gpu_native.slurm"
CANON="${CANON:-/home/phuc/workspace/rl/small_prs/pr001_tinker_mvp/nousnet_pre_multilora_8178f9c/results/code7x_exactinit_canonical}"
NODE="${NODE:?Set NODE to one healthy idle 8-GPU node, e.g. NODE=d2dfac12-018}"
PREFIX="${PREFIX:-repro10_$(date -u +%Y%m%dT%H%M%SZ)}"
[[ -d "${CANON}" ]] || { echo "ERROR: canonical init missing: ${CANON}" >&2; exit 1; }
RUN_CFG_DIR="${REPO_ROOT}/results/reproduction_configs/${PREFIX}"
mkdir -p "${RUN_CFG_DIR}"
cp -f "${CFG_DIR}/base.yaml" "${RUN_CFG_DIR}/base.yaml"
cp -f "${CFG_DIR}/final10_noclip_multi.yaml" "${RUN_CFG_DIR}/multi.yaml"
sed -i "s/^run_name:.*/run_name: ${PREFIX}_multi/" "${RUN_CFG_DIR}/multi.yaml"
for ad in a b c d; do
  cp -f "${CFG_DIR}/final10_noclip_single_${ad}.yaml" "${RUN_CFG_DIR}/single_${ad}.yaml"
  sed -i "s/^run_name:.*/run_name: ${PREFIX}_single_${ad}/" "${RUN_CFG_DIR}/single_${ad}.yaml"
done
BASE_ENV="NOUSNET_DIAG_ENABLED=1,NOUSNET_DIAG_LOSS_TRACE=1,NOUSNET_DIAG_TRACE_ONLY=1,NOUSNET_DIAG_LORA_STEP=0,NOUSNET_INIT_IMPORT_DIR=${CANON}"
submit() {
  local who="$1" cfg="$2" slot="$3" diag_who="$4" jid
  jid=$(sbatch --parsable -w "${NODE}" \
    --export=ALL,CFG="${cfg}",${BASE_ENV},NOUSNET_INIT_IMPORT_SLOT="${slot}",NOUSNET_DIAG_WHO="${diag_who}",NOUSNET_PER_ADAPTER_GRAD_CLIP=0 \
    -J "${PREFIX}-${who}" "${LAUNCHER}")
  printf '%s=%s ' "${who}" "${jid}"
}
printf '%s ' "$(submit multi "${RUN_CFG_DIR}/multi.yaml" '' multi)"
declare -A SLOT=([a]=0 [b]=1 [c]=2 [d]=3)
for ad in a b c d; do
  printf '%s ' "$(submit "s${ad}" "${RUN_CFG_DIR}/single_${ad}.yaml" "${SLOT[$ad]}" "single_${ad}")"
done
printf '\n'
