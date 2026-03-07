#!/bin/bash
#SBATCH --job-name=controlnet_fill50k
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=09:00:00
#SBATCH --output=logs/controlnet_train_%j.log
#SBATCH --error=logs/controlnet_train_%j.err

python Image_process/batch_canny.py --input shared/datasets/coco/canny/images/ --output shared/datasets/coco/canny/edges/ --recursive --workers 16