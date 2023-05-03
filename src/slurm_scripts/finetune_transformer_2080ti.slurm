#!/bin/bash

#SBATCH --job-name=SADTech_roberta_finetune
#SBATCH --mail-user=abhinavp@uw.edu

#SBATCH --account=stf
#SBATCH --partition=gpu-2080ti
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=2:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/mmfs1/gscratch/stf/abhinavp/573-SADTech/src/
#SBATCH --export=all
#SBATCH --output=/mmfs1/home/abhinavp/.logs/%x_%j.out # where STDOUT goes
#SBATCH --error=/mmfs1/home/abhinavp/.logs/%x_%j.err # where STDERR goes

# Modules to use (optional).
# <e.g., module load apptainer>

./run_with_conda.hyak.sh "gpu_SADTech" python -m transformer_lm train -c config/xlm_roberta_twitter_sentiment/rebalance_multiclass_config1.yml
#./run_with_conda.hyak.sh "gpu_SADTech" python finetune_transformer.py -c config/xlm_roberta_twitter_sentiment/rebalance_multiclass_config1.yml
#./run_with_conda.hyak.sh "gpu_SADTech" python finetune_transformer.py -c config/xlm_roberta/autobatch_multiclass_config7.yml
#python finetune_transformer.py -c transformer_configs/xlm_roberta_twitter_sentiment/multiclass_config.yml


#python finetune_transformer.py -c transformer_configs/indic_bert/regression_training_args.yml
#python finetune_transformer.py --help

# ./finetune_transformer.hyak.sh tam debug_10_epochs_2
# ./finetune_transformer.hyak.sh mal debug_2grps_3

# ./finetune_transformer.hyak.sh tam debug_regression_1000epochs
# ./finetune_transformer.hyak.sh mal debug_regression_1000epochs
# ./finetune_transformer.hyak.sh tam_mal_shuffled debug_400epochs
# ./finetune_transformer.hyak.sh mal debug_2grps_4

# disregarded:
#SBATCH --mail-type=END
#SBATCH --open-mode=append # append or truncate
