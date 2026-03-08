#!/bin/bash
#SBATCH --job-name=controlnet_inference_canny
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=logs/controlnet_inference_%j.log
#SBATCH --error=logs/controlnet_inference_%j.err

python Quality_Metrics/inference.py