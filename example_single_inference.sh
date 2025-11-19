#!/bin/bash


set -e  # Exit on any error

# Use 720x1280 resolution for high resolution (use 480x640 for lower and faster generation)
# This script generates all videos for the test dataset
python ./VideoX-Fun/examples/wan2.2/single_predict.py \
    --resolution 720x1280 \
    --model_name ./Models/Wan2.2-I2V-A14B \
    --lora_low ./Models/Lora/10_LargeMixedDatset_wan_14bLow_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors \
    --lora_high ./Models/Lora/10_LargeMixedDatset_wan_14bHigh_f81_LongCaption_StartMatch_run_r128_a128_3obj_Longrun_B4/checkpoint-600.safetensors \
    --config_path ./VideoX-Fun/config/wan2.2/wan_civitai_i2v.yaml \
    --data_csv ./Data/combined_first_frames/0-data.csv \
    --output_path ./output/ffgo_eval \
    --image_path ./Data/fun2.png \
    --caption "High-quality, photorealistic video with cinematic lighting. The scene takes place outdoors in a bright, sunny parking lot in front of a modern white Tesla dealership building featuring large glass windows and red branding accents. Lei Jun, dressed in a formal dark suit with a light blue tie, stands centrally between two parked electric sedans. To his left is a dark blue-grey Xiaomi SU7 with sporty yellow brake calipers, and to his right is a sleek white Tesla Model 3. The camera captures Lei Jun ignoring the dark car on his left and walking purposefully toward the white Tesla Model 3 on his right. Displaying visible admiration, he leans in close to the white vehicle and gently kisses the car, emphasizing his deep affection and preference for the Tesla."