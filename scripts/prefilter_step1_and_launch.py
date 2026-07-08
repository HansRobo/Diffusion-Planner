import json
import os
import subprocess
import time
from pathlib import Path

from diffusion_planner.utils.scene_skip import filter_scene_list
from diffusion_planner.utils.train_utils import openjson

ROOT = Path('/mnt/nvme/Diffusion-Planner-dfp-shared-stack')
OUT = ROOT / 'artifacts' / 'step1_base_skipfiltered_20260626'
OUT.mkdir(parents=True, exist_ok=True)
SRC_TRAIN = Path('/mnt/storage_rdma/diffusion_planner/dataset/20260623_full_sequence/path_list_train.json')
SRC_VALID = Path('/mnt/storage_rdma/diffusion_planner/dataset/20260623_full_sequence/path_list_valid_sft.json')
DST_TRAIN = OUT / 'path_list_train_step1_skipfiltered.json'
DST_VALID = OUT / 'path_list_valid_sft_step1_skipfiltered.json'
LOG = ROOT / 'prefilter_step1_base_node01.log'


def log(msg: str) -> None:
    line = f'[{time.strftime("%Y-%m-%d %H:%M:%S %Z")}] {msg}'
    print(line, flush=True)
    with LOG.open('a') as f:
        f.write(line + '\n')


def load_files(path: Path) -> list[str]:
    data = openjson(str(path))
    return data['files'] if isinstance(data, dict) else data


def write_filtered(src: Path, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        log(f'reuse existing {dst}')
        return
    files = load_files(src)
    log(f'filter start src={src} count={len(files)}')
    filtered = filter_scene_list(files, enabled=True, label=str(src))
    payload = {
        'files': filtered,
        'source': str(src),
        'created_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'skip_filter_applied': True,
        'source_count': len(files),
        'filtered_count': len(filtered),
    }
    tmp = dst.with_suffix(dst.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(payload, f)
    tmp.replace(dst)
    log(f'filter done dst={dst} filtered={len(filtered)} dropped={len(files) - len(filtered)}')


write_filtered(SRC_TRAIN, DST_TRAIN)
write_filtered(SRC_VALID, DST_VALID)

run_name = 'dfp_shared_stack_step1_base_pdms_tier4main_node01_8gpu_tf32'
launch_log = ROOT / 'launch_step1_base_pdms_node01.log'
cmd = (
    'cd /mnt/nvme/Diffusion-Planner-dfp-shared-stack/diffusion_planner && '
    f'RUN_NAME={run_name} PORT=22396 '
    f'TRAIN_LIST={DST_TRAIN} VALID_LIST={DST_VALID} SKIP_FILTER=False '
    'nohup bash ./run_dfp_shared_stack_step1_base_pdms.sh '
    f'> {launch_log} 2>&1 & echo $!'
)
log(f'launch command: {cmd}')
proc = subprocess.run(['bash', '-lc', cmd], text=True, capture_output=True, check=False)
log(f'launch returncode={proc.returncode} stdout={proc.stdout.strip()} stderr={proc.stderr.strip()}')
