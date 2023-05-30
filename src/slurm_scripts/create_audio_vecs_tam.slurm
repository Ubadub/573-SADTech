#!/bin/bash

#SBATCH --job-name=SADTech_create_tam_audio_vecs
#SBATCH --mail-user=abhinavp@uw.edu
#SBATCH --mail-type=ALL

#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --gres=gpu:a40:1
#SBATCH --time=4:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/mmfs1/gscratch/stf/abhinavp/573-SADTech/src/
#SBATCH --export=all
#SBATCH --output=/mmfs1/home/abhinavp/.logs/%x_%j.out # where STDOUT goes
#SBATCH --error=/mmfs1/home/abhinavp/.logs/%x_%j.err # where STDERR goes

# Modules to use (optional).
# <e.g., module load apptainer>

./run_with_conda.hyak.sh "gpu_SADTech" python pipeline_transformers/create_audio_vectors.py -m "Amrrs/wav2vec2-large-xlsr-53-tamil" "vasista22/whisper-tamil-small" "speechbrain/m-ctc-t-large" -d ../data/tam/train_dataset_dict_featurized_audio -o ../data/tam/train_dataset_dict_featurized_audio_new -b 0.05 -l DEBUG -c -1

# disregarded:
#SBATCH --open-mode=append # append or truncate
