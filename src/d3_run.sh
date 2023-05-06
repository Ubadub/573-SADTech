#!/bin/sh

echo "Activating Environment"
source /home2/taraw28/miniconda3/etc/profile.d/conda.sh
conda activate /home2/taraw28/miniconda3/envs/SADTech

python3 src/preprocessing/convert_audio.py