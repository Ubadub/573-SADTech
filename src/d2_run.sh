#!/bin/sh


ENV_PATH=/home2/taraw28/miniconda3/envs/SADTech
DEBUG="${1:-no}"

if [[ "$DEBUG" == "yes" ]]
then
    echo "DEBUG MODE"
    set -euo pipefail
else
    echo "Activating Environment"
    CONDA_PROFILE=$(realpath $(dirname $CONDA_EXE)/../etc/profile.d/conda.sh)

    if [[ ! -f $CONDA_PROFILE ]]
    then
        echo "Conda profile not found. Tried searching at path $CONDA_PROFILE. If this is not where your conda is installed, please edit the script with the correct path. Exiting."
        exit 1
    fi
    source $CONDA_PROFILE
    conda activate $ENV_PATH
fi

owd="$(pwd)"

# Enter src directory
cd $(dirname "$0")

NB_OUTPUT_FILE="nb_output.txt"
TAM_NB_OUTPUT="$(realpath ../outputs/D2/tam/$NB_OUTPUT_FILE)"
MAL_NB_OUTPUT="$(realpath ../outputs/D2/mal/$NB_OUTPUT_FILE)"

TAM_NB_CONFIG="$(realpath config/nb_tam.yml)"
MAL_NB_CONFIG="$(realpath config/nb_mal.yml)"

SCORES_FILE="$(realpath ../results/D2_scores.out)"


# echo "Creating Database"
# python3 run_preprocessing.py tam
# python3 run_preprocessing.py mal

echo "Resetting Output/Results Files"

> $MAL_NB_OUTPUT
> $TAM_NB_OUTPUT
> $SCORES_FILE

echo "Running Baseline: Naive Bayes Classifier"
python -m classifiers $TAM_NB_CONFIG
python -m classifiers $MAL_NB_CONFIG

echo "Running Finetuned Transformer LM Inference - TAMIL"
echo "### TAMIL ###" >> $SCORES_FILE
python -m transformer_lm infer -l tam >> $SCORES_FILE

echo "Running Finetuned Transformer LM Inference - MALAYALAM"
echo "### MALAYALAM ###" >> $SCORES_FILE
python -m transformer_lm infer -l mal >> $SCORES_FILE

# Go back to starting directory
cd $owd

echo "DONE"
