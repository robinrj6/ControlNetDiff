#!/bin/bash
#SBATCH --job-name=controlnet_canny
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=09:00:00
#SBATCH --output=logs/controlnet_canny_%j.log
#SBATCH --error=logs/controlnet_canny_%j.err

python Image_process/batch_canny.py