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

# The current node02 SFT predates this repair and was launched outside Slurm.
# Keep the node drained until that process tree exits naturally, then validate
# the node and return it to service without requiring a second sudo session.
resume_script=/usr/local/sbin/slurm-node02-resume-when-idle
cat >"${resume_script}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
while :; do
  gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^[[:space:]]*$/d' | sort -u | paste -sd, -)
  train_pids=$(pgrep -u ubuntu -f 'train_predictor.py|torch.distributed.run' \
    | sort -u | paste -sd, - || true)
  if [[ -z "${gpu_pids}" && -z "${train_pids}" ]]; then
    break
  fi
  echo "Waiting for unmanaged training to exit: gpu=${gpu_pids:-none} train=${train_pids:-none}"
  sleep 30
done
/etc/slurm/healthcheck.sh
scontrol update NodeName=node02 State=RESUME
echo "node02 passed health checks and was returned to Slurm service"
EOF
chmod 0755 "${resume_script}"

systemctl stop slurm-node02-auto-resume.service 2>/dev/null || true
systemd-run \
  --unit=slurm-node02-auto-resume \
  --property=Type=exec \
  --property=Restart=on-failure \
  --property=RestartSec=60 \
  "${resume_script}"

echo "node02 remains drained while the current SFT runs."
echo "slurm-node02-auto-resume.service will validate and resume it after natural completion."
