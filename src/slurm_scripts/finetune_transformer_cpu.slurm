#!/bin/bash

#SBATCH --job-name=SADTech_roberta_finetune_cpu
#SBATCH --mail-user=abhinavp@uw.edu

#SBATCH --account=stf
#SBATCH --partition=compute-hugemem
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/mmfs1/gscratch/stf/abhinavp/573-SADTech/src/
#SBATCH --export=all
#SBATCH --output=/mmfs1/home/abhinavp/.logs/%x_%j.out # where STDOUT goes
#SBATCH --error=/mmfs1/home/abhinavp/.logs/%x_%j.err # where STDERR goes

# Modules to use (optional).
# <e.g., module load apptainer>

./run_with_conda.hyak.sh "SADTech" python -m transformer_lm train -c config/xlm_roberta_twitter_sentiment/multiclass_config_cpu.yml
#./run_with_conda.hyak.sh "SADTech" python finetune_transformer.py -c transformer_configs/xlm_roberta_twitter_sentiment/multiclass_config_cpu.yml
# ./run_with_conda.hyak.sh "SADTech" python finetune_transformer.py -c transformer_configs/xlm_roberta_twitter_sentiment/multiclass_config_cpu.yml

#./finetune_transformer.hyak.sh -c transformer_configs/xlm_roberta_twitter_sentiment/multiclass_config_cpu.yml
#./finetune_transformer.hyak.sh -c transformer_configs/xlm_roberta_twitter_sentiment/multiclass_config.yml
#./finetune_transformer.hyak.sh -c transformer_configs/indic_bert/regression_training_args.yml
#./finetune_transformer.hyak.sh tam debug_regression_1000epochs
#./finetune_transformer.hyak.sh mal debug_regression_1000epochs
#./finetune_transformer.hyak.sh tam_mal_shuffled debug_400epochs
#./finetune_transformer.hyak.sh mal debug_2grps_4

# disregarded:
#SBATCH --open-mode=append # append or truncate
