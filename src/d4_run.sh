#!/bin/sh


ENV_PATH="/projects/assigned/2223_ling573_group6/573-SADTech/env/573-SADTech"
DEBUG="${1:-False}"

if [[ "$DEBUG" == "True" ]]
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

MODELS_DIR="$(realpath ../outputs/D4/models)"
PRIMARY_TAM_MODEL_DIR=$MODELS_DIR"/primary/tam/"
PRIMARY_MAL_MODEL_DIR=$MODELS_DIR"/primary/mal/"
ADAPTATION_MODEL_DIR=$MODELS_DIR"/adaptation/"

ROOT_SCORES_DIR="$(realpath ../results/D4/)"
PRIMARY_SCORES_DIR=$ROOT_SCORES_DIR"/primary"
ADAPTATION_SCORES_DIR=$ROOT_SCORES_DIR"/adaptation"
PRIMARY_DEVTEST_SCORES_DIR=$PRIMARY_SCORES_DIR"/devtest/"
PRIMARY_EVALTEST_SCORES_DIR=$PRIMARY_SCORES_DIR"/evaltest/"
ADAPTATION_DEVTEST_SCORES_DIR=$ADAPTATION_SCORES_DIR"/devtest/"
ADAPTATION_EVALTEST_SCORES_DIR=$ADAPTATION_SCORES_DIR"/evaltest/"
SCORES_FILE="D4_scores.out"
# PRIMARY_DEVTEST_SCORES_FILE="$(realpath ../results/D4/primary/devtest/D3_scores.out)"

echo "Resetting Output/Results Files"

echo -n "" > "$PRIMARY_DEVTEST_SCORES_DIR$SCORES_FILE"
echo -n "" > "$PRIMARY_EVALTEST_SCORES_DIR$SCORES_FILE"
echo -n "" > "$ADAPTATION_EVALTEST_SCORES_DIR$SCORES_FILE"
echo -n "" > "$ADAPTATION_DEVTEST_SCORES_DIR$SCORES_FILE"

echo "###### Primary- Tamil, Logistic Regression, Ridge Regression Feature Selection, Text Only ######" >> "$PRIMARY_DEVTEST_SCORES_DIR$SCORES_FILE"
echo "###### Primary- Tamil, Logistic Regression, Ridge Regression Feature Selection, Text Only ######" >> "$PRIMARY_EVALTEST_SCORES_DIR$SCORES_FILE"
python inference.py --config-name=inference lang=tam +pipeline_path="$PRIMARY_TAM_MODEL_DIR" '+outputs.sub_dir=primary' '+results.sub_dir=primary' hydra.verbose=$DEBUG

echo -e "\n###### Primary- Malayalam, Logistic Regression, Ridge Regression Feature Selection, Text Only ######" >> "$PRIMARY_DEVTEST_SCORES_DIR$SCORES_FILE"
echo -e "\n###### Primary- Malayalam, Logistic Regression, Ridge Regression Feature Selection, Text Only ######" >> "$PRIMARY_EVALTEST_SCORES_DIR$SCORES_FILE"
python inference.py --config-name=inference lang=mal +pipeline_path="$PRIMARY_MAL_MODEL_DIR" '+outputs.sub_dir=primary' '+results.sub_dir=primary' hydra.verbose=$DEBUG

echo "###### Adaptation (combined), Logistic Regression, Ridge Regression Feature Selection, Text + Audio  ######" >> "$ADAPTATION_DEVTEST_SCORES_DIR$SCORES_FILE"
echo "###### Adaptation (combined), Logistic Regression, Ridge Regression Feature Selection, Text + Audio  ######" >> "$ADAPTATION_EVALTEST_SCORES_DIR$SCORES_FILE"
python inference.py --config-name=inference lang=tam_mal_shuffled +pipeline_path="$ADAPTATION_MODEL_DIR" '+outputs.sub_dir=adaptation' '+results.sub_dir=adaptation' hydra.verbose=$DEBUG


# Go back to starting directory
cd $owd

echo "DONE"
