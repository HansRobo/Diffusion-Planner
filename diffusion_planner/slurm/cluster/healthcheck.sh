#!/usr/bin/env bash

# Slurm runs this only while a node is IDLE. Keep optional filesystems out of
# the global gate; individual jobs must preflight every dataset they require.
set -u

fail() {
  echo "DRAIN: $*"
  exit 1
}

gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
[[ "${gpu_count}" -eq 8 ]] || fail "only ${gpu_count} GPUs detected (expected 8)"

mountpoint -q /mnt/nvme || fail "NVMe scratch /mnt/nvme not mounted"

case "$(hostname -s)" in
  node02)
    mountpoint -q /mnt/storage_rdma || fail "RDMA dataset mount /mnt/storage_rdma not mounted"
    ;;
  node01)
    # node01 supports both Lustre and RDMA workloads. A job that needs either
    # mount checks its own input paths before allocating workers.
    if ! mountpoint -q /mnt/share_drive && ! mountpoint -q /mnt/storage_rdma; then
      fail "neither /mnt/share_drive nor /mnt/storage_rdma is mounted"
    fi
    ;;
  *)
    fail "no mount policy configured for $(hostname -s)"
    ;;
esac

# An IDLE Slurm node must not have an unmanaged CUDA workload. This catches
# direct torchrun launches before Slurm can allocate the same GPUs again.
gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
  | sed '/^[[:space:]]*$/d' | sort -u | paste -sd, -)
[[ -z "${gpu_pids}" ]] || fail "unmanaged GPU compute processes detected: ${gpu_pids}"

free_mem_kib=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
[[ "${free_mem_kib}" -ge 10485760 ]] \
  || fail "low memory ($((free_mem_kib / 1024)) MiB available)"

exit 0
