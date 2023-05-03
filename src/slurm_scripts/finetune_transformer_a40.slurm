#!/bin/bash

#SBATCH --job-name=SADTech_roberta_finetune
#SBATCH --mail-user=abhinavp@uw.edu

#SBATCH --account=stf
#SBATCH --partition=ckpt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtx6k:1
#SBATCH --time=4:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/mmfs1/gscratch/stf/abhinavp/573-SADTech/src/
#SBATCH --export=all
#SBATCH --output=/mmfs1/home/abhinavp/.logs/%x_%j.out # where STDOUT goes
#SBATCH --error=/mmfs1/home/abhinavp/.logs/%x_%j.err # where STDERR goes

# Modules to use (optional).
# <e.g., module load apptainer>

./run_with_conda.hyak.sh "gpu_SADTech" python -m transformer_lm train -c config/indic_bert/rebalance_multiclass_config2.yml
#./run_with_conda.hyak.sh "gpu_SADTech" python transformer_lm/finetune_transformer.py -c config/indic_bert/rebalance_multiclass_config2.yml
#./run_with_conda.hyak.sh "gpu_SADTech" python finetune_transformer.py -c config/xlm_roberta_twitter_sentiment/rebalance_multiclass_config1.yml
#./run_with_conda.hyak.sh "gpu_SADTech" python finetune_transformer.py -c config/xlm_roberta/multiclass_config7.yml

#conda deactivate
#echo "Done."

#./finetune_transformer.hyak.sh -c transformer_configs/xlm_roberta/multiclass_config6.yml
#./finetune_transformer.hyak.sh -c transformer_configs/xlm_roberta_twitter_sentiment/regression_config.yml
#./finetune_transformer.hyak.sh -c transformer_configs/xlm_roberta_twitter_sentiment/multiclass_config.yml
#./finetune_transformer.hyak.sh -c transformer_configs/indic_bert/regression_training_args.yml
#./finetune_transformer.hyak.sh tam debug_regression_1000epochs
#./finetune_transformer.hyak.sh mal debug_regression_1000epochs
#./finetune_transformer.hyak.sh tam_mal_shuffled debug_400epochs
#./finetune_transformer.hyak.sh mal debug_2grps_4

# disregarded:
#SBATCH --open-mode=append # append or truncate
