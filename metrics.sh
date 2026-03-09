#!/bin/bash
#SBATCH --job-name=controlnet_metrics
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/controlnet_metrics_%j.log
#SBATCH --error=logs/controlnet_metrics_%j.err


python Quality_Metrics/qualityMetrics.py 