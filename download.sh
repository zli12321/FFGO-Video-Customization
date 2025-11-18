#!/bin/bash

# Script to download Hugging Face models
# Make sure huggingface_hub[cli] is installed before running

set -e  # Exit on any error

huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./Models/Wan2.2-I2V-A14B


huggingface-cli download Video-Customization/FFGO-Lora-Adapter --local-dir ./Models/Lora