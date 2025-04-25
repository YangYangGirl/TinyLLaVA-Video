#!/bin/bash
#SBATCH --account=OD-227441
#SBATCH --job-name=train_qwen2_sft
#SBATCH --output=slurm_outputs/train_qwen2_sft_nba.out
#SBATCH --error=slurm_outputs/train_qwen2_sft_nba.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=500G
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32

# === Setup your environment ===
conda activate video_ruler

# === Run your training script ===
# Assign the arguments to variables
VIDEO_DATA_PATH="${HF_HOME}/data/nba_ov"
VIDEO_PATH="${HF_HOME}/data/nba_ov/0_60_s_nba/nba_videos_meta_reason_train_converted.json"
LLM_VERSION=${HF_HOME}/checkpoints/Qwen2.5-3B # llm path
VT_VERSION=${HF_HOME}/checkpoints/siglip-so400m-patch14-384 # vision tower path
VT_VERSION2=""
CN_VERSION=groupresampler
CONV_VERSION=qwen2_base
VERSION=base
TRAIN_RECIPE=common
MODEL_MAX_LENGTH=3072
NUM_FRAME=16
NUM_QUERY=512
GROUP=16

VT_VARIANT="${VT_VERSION##*/}"
LLM_VARIANT="${LLM_VERSION##*/}"

deepspeed --include localhost:0,1,2,3 --master_port 29501 tinyllava/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --video_data_path  $VIDEO_DATA_PATH \
    --video_folder $VIDEO_PATH \
    --is_multimodal True \
    --conv_version $CONV_VERSION \
    --model_name_or_path $LLM_VERSION \
    --vision_tower $VT_VERSION \
    --vision_tower2 "$VT_VERSION2" \
    --connector_type $CN_VERSION \
    --num_frames $NUM_FRAME \
    --num_queries $NUM_QUERY \
    --group $GROUP \
    --mm_vision_select_layer -2 \
    --image_aspect_ratio square \
    --attn_implementation flash_attention_2 \
    --bf16 True \
    --training_recipe $TRAIN_RECIPE \
    --tune_type_llm full \
    --tune_type_vision_tower frozen \
    --tune_vision_tower_from_layer 0 \
    --tune_type_connector full \
    --group_by_modality_length False \
    --pretrained_model_path ${HF_HOME}/checkpoints/TinyLLaVA-Video-Qwen2.5-3B-Group-16-512 \
    --output_dir ${HF_HOME}/checkpoints/sft/tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-nba \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --tokenizer_use_fast False \
    --run_name tiny-llava-${LLM_VARIANT}-${VT_VARIANT}-${VERSION}-finetune
