#!/bin/bash
export PYTHONPATH=/home/ai_center/ai_users/alonmardi/projects/Grounded-SAM-2:$PYTHONPATH
PYTHON_EXEC="/home/ai_center/ai_users/alonmardi/miniconda/envs/gd_sam/bin/python"

#SBATCH --job-name=gaze_extraction
#SBATCH --output=gaze_extraction_%j.out
#SBATCH --error=gaze_extraction_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4  # Adjust as needed
#SBATCH --mem=32G          # Adjust as needed
#SBATCH --gres=gpu:1       # Request 1 GPU, as the Python script uses CUDA
#SBATCH --time=01:00:00    # Adjust wall clock time limit as needed (HH:MM:SS)

# --- Execute the Python script ---
# The Python script is located at gazefollow/extract_gaze_target_desc_refactored.py
echo "Starting Python script execution..."
$PYTHON_EXEC gazefollow/extract_gaze_target_desc_refactored.py
echo "Python script execution finished." 