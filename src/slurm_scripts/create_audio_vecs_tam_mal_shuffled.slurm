#!/bin/bash

#SBATCH --job-name=SADTech_create_tam_mal_audio_vecs
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

./run_with_conda.hyak.sh "gpu_SADTech" python pipeline_transformers/create_audio_vectors.py -m "facebook/wav2vec2-large-xlsr-53" "microsoft/unispeech-large-1500h-cv" -d ../data/tam_mal_shuffled/train_dataset_dict_featurized_audio2 -o ../data/tam_mal_shuffled/train_dataset_dict_featurized_audio3 -b 0.01 -l DEBUG -c -2 -3
# ./run_with_conda.hyak.sh "gpu_SADTech" python pipeline_transformers/create_audio_vectors.py -m "facebook/wav2vec2-large-xlsr-53" "microsoft/unispeech-large-1500h-cv" -d ../data/tam_mal_shuffled/train_dataset_dict_featurized_audio -o ../data/tam_mal_shuffled/train_dataset_dict_featurized_audio2 -b 0.01 -l DEBUG -c -4
# ./run_with_conda.hyak.sh "gpu_SADTech" python pipeline_transformers/create_audio_vectors.py -m "facebook/wav2vec2-large-xlsr-53" "microsoft/unispeech-large-1500h-cv" -d ../data/tam_mal_shuffled/train_dataset_dict_featurized_audio -o ../data/tam_mal_shuffled/train_dataset_dict_featurized_audio2 -b 0.01 -l DEBUG -c -3
# ./run_with_conda.hyak.sh "gpu_SADTech" python pipeline_transformers/create_audio_vectors.py -m "facebook/wav2vec2-large-xlsr-53" "microsoft/unispeech-large-1500h-cv" -d ../data/tam_mal_shuffled/train_dataset_dict_featurized_audio -o ../data/tam_mal_shuffled/train_dataset_dict_featurized_audio2 -b 0.01 -l DEBUG -c -2
# ./run_with_conda.hyak.sh "gpu_SADTech" python pipeline_transformers/create_audio_vectors.py -m "facebook/wav2vec2-large-xlsr-53" "microsoft/unispeech-large-1500h-cv" -d ../data/tam_mal_shuffled/train_dataset_dict_featurized_audio -o ../data/tam_mal_shuffled/train_dataset_dict_featurized_audio2 -b 0.01 -l DEBUG -c -1
# ./run_with_conda.hyak.sh "gpu_SADTech" python pipeline_transformers/create_audio_vectors.py -m "facebook/wav2vec2-large-xlsr-53" "microsoft/unispeech-large-1500h-cv" -d ../data/tam_mal_shuffled/train_dataset_dict_audio -o ../data/tam_mal_shuffled/train_dataset_dict_featurized_audio -b 0.01 -l DEBUG -c -1 -2
# ./run_with_conda.hyak.sh "gpu_SADTech" python pipeline_transformers/create_audio_vectors.py -m "facebook/wav2vec2-large-xlsr-53" "microsoft/unispeech-large-1500h-cv" -d ../data/tam_mal_shuffled/train_dataset_dict_audio -o ../data/tam_mal_shuffled/train_dataset_dict_featurized_audio -b 0.01 -l DEBUG -c -1 -2 -3 -4

# disregarded:
#SBATCH --open-mode=append # append or truncate
