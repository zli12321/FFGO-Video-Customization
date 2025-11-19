#!/bin/bash


set -e  # Exit on any error

# Use 720x1280 resolution for high resolution (use 480x640 for lower and faster generation)
# This script generates all videos for the test dataset
python ./VideoX-Fun/examples/wan2.2/batch_predict-v1.py \
    --resolution 480x640 \
    --model_name ./Models/Wan2.2-I2V-A14B \
    --lora_low ./Models/Lora/10_LargeMixedDatset_wan_14bLow_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors \
    --lora_high ./Models/Lora/10_LargeMixedDatset_wan_14bHigh_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors \
    --config_path ./VideoX-Fun/config/wan2.2/wan_civitai_i2v.yaml \
    --data_csv ./Data/combined_first_frames/0-data.csv \
    --output_path ./output/ffgo_eval