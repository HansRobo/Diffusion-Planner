#!/usr/bin/env bash
set -euo pipefail

# Archive the auditable replay contract and remove only the large replay arrays
# after a completed training/selection run exists.  Dry-run is the default.
#
# Usage:
#   archive_prune_replay_cache.sh CACHE_RUN EVIDENCE_RUN EXPECTED_TOTAL [--execute]
#
# CACHE_RUN owns replay_buffer/. EVIDENCE_RUN must contain final_summary.json
# or selected_checkpoint.json plus a recoverable checkpoint.  They are often
# the same directory for an uninterrupted run and different for split
# mining/replay runs.

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 CACHE_RUN EVIDENCE_RUN EXPECTED_TOTAL [--execute]" >&2
  exit 2
fi

CACHE_RUN=$(realpath -e "$1")
EVIDENCE_RUN=$(realpath -e "$2")
EXPECTED_TOTAL=$3
MODE=${4:-"--dry-run"}

[[ ${EXPECTED_TOTAL} =~ ^[1-9][0-9]*$ ]] || {
  echo "EXPECTED_TOTAL must be a positive integer" >&2
  exit 2
}
[[ ${MODE} == "--dry-run" || ${MODE} == "--execute" ]] || {
  echo "fourth argument must be --execute" >&2
  exit 2
}

ROOT=/mnt/nvme/wangbin/Diffusion-Planner-t4-main
OUT=${ROOT}/outputs
case "${CACHE_RUN}/" in
  "${OUT}/"*) ;;
  *) echo "refusing cache outside ${OUT}: ${CACHE_RUN}" >&2; exit 1 ;;
esac
case "${EVIDENCE_RUN}/" in
  "${OUT}/"*) ;;
  *) echo "refusing evidence outside ${OUT}: ${EVIDENCE_RUN}" >&2; exit 1 ;;
esac

CACHE=${CACHE_RUN}/replay_buffer
[[ -d ${CACHE} ]] || { echo "missing ${CACHE}" >&2; exit 1; }
[[ -f ${CACHE_RUN}/effective_config.json ]] || {
  echo "missing cache effective_config.json" >&2
  exit 1
}
[[ -f ${CACHE_RUN}/provenance.json ]] || {
  echo "missing cache provenance.json" >&2
  exit 1
}
if [[ ! -f ${EVIDENCE_RUN}/final_summary.json && ! -f ${EVIDENCE_RUN}/selected_checkpoint.json ]]; then
  echo "evidence run has neither final_summary.json nor selected_checkpoint.json" >&2
  exit 1
fi

CHECKPOINT=""
if [[ -f ${EVIDENCE_RUN}/selected_checkpoint.json ]]; then
  selected=$(jq -er '.checkpoint' "${EVIDENCE_RUN}/selected_checkpoint.json")
  [[ ${selected} != */* ]] || {
    echo "selected checkpoint must be a basename, got ${selected}" >&2
    exit 1
  }
  [[ -s ${EVIDENCE_RUN}/${selected} ]] && CHECKPOINT=${EVIDENCE_RUN}/${selected}
fi
if [[ -z ${CHECKPOINT} ]]; then
  for candidate in \
    "${EVIDENCE_RUN}/best_train.pth" \
    "${EVIDENCE_RUN}/best_model_awr_retained_e004_a0p05.pth"; do
    if [[ -s ${candidate} ]]; then
      CHECKPOINT=${candidate}
      break
    fi
  done
fi
if [[ -z ${CHECKPOINT} && -f ${EVIDENCE_RUN}/final_summary.json ]]; then
  # Early launch/smoke runs predate best_train.pth but still record their
  # selected validation epoch.  Accept that evidence only when the exact
  # checkpoint named by the summary exists; epoch zero is the immutable base.
  best_eval_epoch=$(jq -er '.best_eval_epoch | numbers' \
    "${EVIDENCE_RUN}/final_summary.json" 2>/dev/null || true)
  if [[ ${best_eval_epoch} =~ ^[0-9]+$ ]]; then
    if [[ ${best_eval_epoch} -eq 0 ]]; then
      candidate=${EVIDENCE_RUN}/base_model/best_model.pth
    else
      printf -v checkpoint_name 'epoch_%03d.pth' "${best_eval_epoch}"
      candidate=${EVIDENCE_RUN}/${checkpoint_name}
    fi
    [[ -s ${candidate} ]] && CHECKPOINT=${candidate}
  fi
fi
[[ -n ${CHECKPOINT} ]] || {
  echo "no recoverable selected/best checkpoint in evidence run" >&2
  exit 1
}

# A replay reader includes the absolute cache path in its command line. Never
# prune while any process still advertises that path. Read /proc directly so
# the checker does not accidentally match its own grep/awk command line.
users_file=/tmp/awr_replay_cache_users.$$
: > "${users_file}"
for cmdline in /proc/[0-9]*/cmdline; do
  pid=${cmdline#/proc/}
  pid=${pid%/cmdline}
  [[ ${pid} == "$$" || ${pid} == "${PPID}" ]] && continue
  command=$(tr '\0' ' ' 2>/dev/null < "${cmdline}" || true)
  [[ ${command} == *"${CACHE}"* ]] || continue
  printf '%s %s\n' "${pid}" "${command}" >> "${users_file}"
done
if [[ -s ${users_file} ]]; then
  echo "refusing cache referenced by a running process:" >&2
  cat "${users_file}" >&2
  rm -f "${users_file}"
  exit 1
fi
rm -f "${users_file}"

# A later refresh may link its immutable encoder/context arrays to this cache.
# Deleting the source would leave a completed downstream manifest pointing at
# dangling files even when no process currently has them open.
dependencies_file=/tmp/awr_replay_cache_dependencies.$$
: > "${dependencies_file}"
while IFS= read -r link; do
  target=$(readlink -f "${link}" 2>/dev/null || true)
  case "${target}" in
    "${CACHE}/"*) printf '%s -> %s\n' "${link}" "${target}" >> "${dependencies_file}" ;;
  esac
done < <(find "${OUT}" -type l -print)
if [[ -s ${dependencies_file} ]]; then
  echo "refusing cache with downstream shared-array dependencies:" >&2
  cat "${dependencies_file}" >&2
  rm -f "${dependencies_file}"
  exit 1
fi
rm -f "${dependencies_file}"

# Safe retry after a previous successful prune (for example, if a downstream
# cycle launcher failed after rotation but before starting the next process).
if [[ -f ${CACHE}/PRUNED.json && -f ${CACHE_RUN}/replay_buffer_archive/archive_summary.json ]]; then
  archived_count=$(jq -er '.scene_count' "${CACHE}/PRUNED.json")
  archived_ranks=$(jq -er '.rank_count' "${CACHE}/PRUNED.json")
  pruned=$(jq -er '.pruned' "${CACHE}/PRUNED.json")
  bulk_count=$(find "${CACHE}" -maxdepth 2 \
    \( -type f -o -type l \) \
    \( -name '*.npy' -o -name 'scene_paths.jsonl' -o -name 'expected_scene_paths.jsonl' \) | wc -l)
  manifest_count=$(find "${CACHE}" -maxdepth 2 -type f -name manifest.json | wc -l)
  if [[ ${pruned} == "true" && ${archived_count} -eq ${EXPECTED_TOTAL} \
        && ${archived_ranks} -eq 8 && ${bulk_count} -eq 0 \
        && ${manifest_count} -eq 8 ]]; then
    echo "cache already pruned and verified: ${CACHE}"
    exit 0
  fi
  echo "inconsistent prior prune marker; refusing retry" >&2
  exit 1
fi

rank_count=0
total=0
for rank in $(seq 0 7); do
  printf -v rank_name 'rank_%04d' "${rank}"
  rank_dir=${CACHE}/${rank_name}
  manifest=${rank_dir}/manifest.json
  paths=${rank_dir}/scene_paths.jsonl
  [[ -f ${manifest} && -f ${paths} ]] || {
    echo "rank ${rank} lacks manifest or scene_paths.jsonl" >&2
    exit 1
  }
  count=$(jq -er '.scene_count' "${manifest}")
  lines=$(wc -l < "${paths}")
  [[ ${count} -eq ${lines} ]] || {
    echo "rank ${rank}: manifest=${count}, paths=${lines}" >&2
    exit 1
  }
  total=$((total + count))
  rank_count=$((rank_count + 1))
done
[[ ${rank_count} -eq 8 && ${total} -eq ${EXPECTED_TOTAL} ]] || {
  echo "cache contract mismatch: ranks=${rank_count}, total=${total}, expected=${EXPECTED_TOTAL}" >&2
  exit 1
}

ARCHIVE=${CACHE_RUN}/replay_buffer_archive
mkdir -p "${ARCHIVE}/manifests"
cp -f "${CACHE_RUN}/effective_config.json" "${ARCHIVE}/effective_config.json"
cp -f "${CACHE_RUN}/provenance.json" "${ARCHIVE}/provenance.json"
for rank in $(seq 0 7); do
  printf -v rank_name 'rank_%04d' "${rank}"
  cp -f "${CACHE}/${rank_name}/manifest.json" "${ARCHIVE}/manifests/${rank_name}.json"
done
[[ ! -f ${EVIDENCE_RUN}/final_summary.json ]] || \
  cp -f "${EVIDENCE_RUN}/final_summary.json" "${ARCHIVE}/evidence_final_summary.json"
[[ ! -f ${EVIDENCE_RUN}/selected_checkpoint.json ]] || \
  cp -f "${EVIDENCE_RUN}/selected_checkpoint.json" "${ARCHIVE}/evidence_selected_checkpoint.json"

find "${CACHE}" -maxdepth 2 -type f -printf '%P\t%s\t%b\t%T@\n' \
  | LC_ALL=C sort > "${ARCHIVE}/file_inventory.tsv"
sha256sum "${ARCHIVE}/effective_config.json" "${ARCHIVE}/provenance.json" \
  "${ARCHIVE}"/manifests/*.json > "${ARCHIVE}/contract_sha256.txt"
checkpoint_sha256=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')
physical_before=$(du -sB1 "${CACHE}" | awk '{print $1}')
jq -n \
  --arg cache_run "${CACHE_RUN}" \
  --arg evidence_run "${EVIDENCE_RUN}" \
  --arg checkpoint "${CHECKPOINT}" \
  --arg checkpoint_sha256 "${checkpoint_sha256}" \
  --argjson scene_count "${total}" \
  --argjson rank_count "${rank_count}" \
  --argjson physical_bytes_before "${physical_before}" \
  '{schema_version:1, cache_run:$cache_run, evidence_run:$evidence_run,
    recoverable_checkpoint:$checkpoint, checkpoint_sha256:$checkpoint_sha256,
    scene_count:$scene_count, rank_count:$rank_count,
    physical_bytes_before:$physical_bytes_before,
    retained:["effective_config.json","provenance.json","8 rank manifests",
              "file_inventory.tsv","contract_sha256.txt","checkpoint evidence"]}' \
  > "${ARCHIVE}/archive_summary.json"

echo "verified ${total} groups across ${rank_count} ranks"
echo "recoverable checkpoint: ${CHECKPOINT} (${checkpoint_sha256})"
echo "cache physical bytes before prune: ${physical_before}"
echo "archive: ${ARCHIVE}"

if [[ ${MODE} != "--execute" ]]; then
  echo "dry-run complete; pass --execute only after the cache is no longer needed"
  exit 0
fi

# Keep rank manifests in place.  Remove only regenerable bulk arrays and the
# full path index after all checks and archival writes have succeeded.
chmod u+w "${CACHE}"
find "${CACHE}" -mindepth 1 -maxdepth 1 -type d -exec chmod u+w {} +
find "${CACHE}" -maxdepth 2 -type f \
  \( -name '*.npy' -o -name 'scene_paths.jsonl' -o -name 'expected_scene_paths.jsonl' \) -exec chmod u+w {} +
find "${CACHE}" -maxdepth 2 \
  \( -type f -o -type l \) \
  \( -name '*.npy' -o -name 'scene_paths.jsonl' -o -name 'expected_scene_paths.jsonl' \) -delete

physical_after=$(du -sB1 "${CACHE}" | awk '{print $1}')
jq --argjson physical_bytes_after "${physical_after}" \
  '. + {pruned:true, physical_bytes_after:$physical_bytes_after}' \
  "${ARCHIVE}/archive_summary.json" > "${ARCHIVE}/archive_summary.json.tmp"
mv "${ARCHIVE}/archive_summary.json.tmp" "${ARCHIVE}/archive_summary.json"
cp -f "${ARCHIVE}/archive_summary.json" "${CACHE}/PRUNED.json"
echo "prune complete: physical bytes ${physical_before} -> ${physical_after}"
