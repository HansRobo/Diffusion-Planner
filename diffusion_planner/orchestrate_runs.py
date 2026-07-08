#!/usr/bin/env python3
import os
import sys
import time
import logging
import subprocess

# Set up logging to print to both console and a log file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("orchestrator.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Ensure working directory is the script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# Configurations to run sequentially after C1_mild finishes
RUNS = [
    {
        "name": "C7_normal_anchor",
        "exp_name": "ratio_experiment_C7_normal_anchor",
        "train_set": "/home/ubuntu/Diffusion-Planner-Meta-Repository/dataset/ratio_experiment/C7_normal_anchor/path_list_train.json",
        "valid_set": "/mnt/nvme/dataset/basic_dataset/path_list_valid_sft_balanced.json"
    },
    {
        "name": "C8_turn_lane_boost",
        "exp_name": "ratio_experiment_C8_turn_lane_boost",
        "train_set": "/home/ubuntu/Diffusion-Planner-Meta-Repository/dataset/ratio_experiment/C8_turn_lane_boost/path_list_train.json",
        "valid_set": "/mnt/nvme/dataset/basic_dataset/path_list_valid_sft_balanced.json"
    }
]

COOLDOWN_SECONDS = 120  # 2 minutes cooldown
POLL_INTERVAL = 30      # Check status every 30 seconds
GPU_UTIL_THRESHOLD = 10 # GPU utilization under 10% is considered idle

def is_process_running(pattern):
    """Check if any running process cmdline contains the given pattern (excluding our own PID)."""
    try:
        result = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        if result.returncode == 0:
            pids = [pid.strip() for pid in result.stdout.strip().split("\n") if pid.strip().isdigit()]
            my_pid = str(os.getpid())
            pids = [pid for pid in pids if pid != my_pid]
            return len(pids) > 0
        return False
    except Exception as e:
        logging.warning(f"Error checking process status for pattern '{pattern}': {e}")
        return False

def get_max_gpu_utilization():
    """Query nvidia-smi for the maximum GPU utilization percentage across all GPUs."""
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        utils = [int(line.strip()) for line in res.stdout.strip().split("\n") if line.strip().isdigit()]
        return max(utils) if utils else 0
    except Exception as e:
        logging.warning(f"Failed to query GPU utilization via nvidia-smi: {e}")
        # Fallback to 0 if nvidia-smi query fails to avoid blocking the queue permanently
        return 0

def wait_for_idle(process_pattern, cooldown_seconds, poll_interval):
    """
    Waits until no processes matching process_pattern are running,
    AND max GPU utilization is below the idle threshold for a sustained cooldown period.
    """
    logging.info(f"Monitoring system. Waiting for process pattern '{process_pattern}' to exit and GPU to become idle.")
    idle_since = None
    
    while True:
        process_active = is_process_running(process_pattern)
        max_gpu_util = get_max_gpu_utilization()
        
        is_idle = (not process_active) and (max_gpu_util < GPU_UTIL_THRESHOLD)
        
        if is_idle:
            if idle_since is None:
                idle_since = time.time()
                logging.info(
                    f"Idle state detected (Process running: {process_active}, Max GPU util: {max_gpu_util}%). "
                    f"Starting {cooldown_seconds}s cooldown verification..."
                )
            else:
                elapsed = time.time() - idle_since
                logging.info(f"System remains idle. Cooldown elapsed: {elapsed:.1f}s / {cooldown_seconds}s.")
                if elapsed >= cooldown_seconds:
                    logging.info("Cooldown completed successfully. System is confirmed idle.")
                    break
        else:
            if idle_since is not None:
                logging.info(
                    f"System became busy again (Process running: {process_active}, Max GPU util: {max_gpu_util}%). "
                    f"Resetting cooldown timer."
                )
                idle_since = None
            else:
                logging.info(f"System busy (Process active: {process_active}, Max GPU util: {max_gpu_util}%). Checking again in {poll_interval}s.")
        
        time.sleep(poll_interval)

def main():
    logging.info("Starting training run orchestrator.")
    
    # Step 1: Wait for C1_mild to finish
    logging.info("Phase 1: Waiting for C1_mild training run to complete...")
    # Monitor train_predictor.py processes containing C1_mild
    c1_pattern = "train_predictor.py.*C1_mild"
    wait_for_idle(c1_pattern, COOLDOWN_SECONDS, POLL_INTERVAL)
    
    # Step 2: Run C2, C3, and C4 consecutively
    logging.info("Phase 2: Initiating sequential runs.")
    for run in RUNS:
        name = run["name"]
        exp_name = run["exp_name"]
        train_set = run["train_set"]
        valid_set = run["valid_set"]
        
        cmd = ["./train_run.sh", exp_name, train_set, valid_set]
        logging.info(f"============================================================")
        logging.info(f"Starting run for configuration: {name}")
        logging.info(f"Command: {' '.join(cmd)}")
        logging.info(f"============================================================")
        
        # Run command and let output stream to standard outputs (so it goes to screen/nohup)
        # Note: train_run.sh also logs to /mnt/nvme/training_result/<TIME>_<exp_name>/train_log.txt
        start_time = time.time()
        result = subprocess.run(cmd, check=False)
        duration = time.time() - start_time
        
        if result.returncode != 0:
            logging.error(
                f"Run {name} FAILED with return code {result.returncode} after {duration:.1f}s. "
                f"Aborting sequence as requested."
            )
            sys.exit(result.returncode)
            
        logging.info(f"Run {name} COMPLETED successfully in {duration:.1f}s.")
        
        # Apply cooldown before starting the next run to allow the system / GPU VRAM to reset
        logging.info(f"Applying {COOLDOWN_SECONDS}s cooldown before the next run...")
        time.sleep(COOLDOWN_SECONDS)
        
    logging.info("All scheduled runs finished successfully!")

if __name__ == "__main__":
    main()
