#!/usr/bin/env bash

# Run as root on node02 after staging controller-slurm.conf and healthcheck.sh
# in the same directory as this script. Reloading slurmd does not stop jobs.
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run with sudo: sudo $0" >&2
  exit 2
fi
if [[ "$(hostname -s)" != "node02" ]]; then
  echo "Refusing to install node02 policy on $(hostname -s)" >&2
  exit 2
fi

src_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
controller_conf=${src_dir}/controller-slurm.conf
healthcheck=${src_dir}/healthcheck.sh
[[ -r "${controller_conf}" ]] || { echo "Missing ${controller_conf}" >&2; exit 2; }
[[ -r "${healthcheck}" ]] || { echo "Missing ${healthcheck}" >&2; exit 2; }
grep -q '^UnkillableStepTimeout=300$' "${controller_conf}" \
  || { echo "Controller config lacks UnkillableStepTimeout=300" >&2; exit 2; }

stamp=$(date +%Y%m%d-%H%M%S)
cp -a /etc/slurm/slurm.conf "/etc/slurm/slurm.conf.bak.${stamp}"
cp -a /etc/slurm/healthcheck.sh "/etc/slurm/healthcheck.sh.bak.${stamp}"
install -o root -g root -m 0644 "${controller_conf}" /etc/slurm/slurm.conf
install -o root -g root -m 0755 "${healthcheck}" /etc/slurm/healthcheck.sh
systemctl reload slurmd

echo "Installed synchronized Slurm config and node-aware health check."
echo "The node intentionally remains drained until all unmanaged GPU processes exit."
