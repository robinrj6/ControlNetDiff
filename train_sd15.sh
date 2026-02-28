#!/bin/bash
#SBATCH --job-name=controlnet_fill50k
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=logs/controlnet_train_%j.log
#SBATCH --error=logs/controlnet_train_%j.err

export MODEL_DIR="/home/hpc/rlvl/rlvl165v/Desktop/diffusers/examples/controlnet/shared/models/sd15"
export DATASET_DIR="/home/hpc/rlvl/rlvl165v/Desktop/diffusers/examples/controlnet/shared/datasets/fill50k/extracted/"
export OUTPUT_DIR="/home/hpc/rlvl/rlvl165v/Desktop/diffusers/examples/controlnet/output/"

export HF_HOME="/home/woody/rlvl/rlvl165v/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_HUB_CACHE="$HF_HOME/hub"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

accelerate launch train_controlnet.py \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --train_data_dir="$DATASET_DIR" \
 --image_column="image" \
 --conditioning_image_column="conditioning" \
 --caption_column="text" \
 --resolution=512 \
 --learning_rate=1e-5 \
 --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png" \
 --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
 --train_batch_size=1 \
 --gradient_accumulation_steps=4 \
 --max_train_steps=50 \
 --checkpointing_steps=10 \
 --validation_steps=10 \
 --gradient_checkpointing \
 --use_8bit_adam \
 --enable_xformers_memory_efficient_attention \
 --set_grads_to_none
