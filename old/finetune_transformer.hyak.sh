#!/bin/bash

# set -euxo pipefail # automatically exit if any program errors, prohibit undefined variables
# set -euo pipefail # automatically exit if any program errors, prohibit undefined variables
set -eo pipefail # automatically exit if any program errors, prevent pipeline errors from being masked

# if you install anaconda in a different directory, change the following line to
# CONDA_PROFILE=path_to_anaconda3/anaconda3/etc/profile.d/conda.sh
CONDA_PROFILE=$(realpath /mmfs1/gscratch/stf/abhinavp/miniconda3/etc/profile.d/conda.sh)

ENV_NAME="gpu_SADTech"
#ENV_NAME="${2:-gpu_SADTech}"

TRAIN_LANG="${1:-tam}"
RUN_ID="${2:-debug}"

conda_env_exists_name(){
    conda env list | grep -E "^${@}\b" >/dev/null 2>/dev/null
}

# echo "Using prefix: $ENV_PREFIX"
echo "Using environment name: $ENV_NAME"

if [[ ! -f $CONDA_PROFILE ]] 
then
    echo "Conda profile not found at $CONDA_PROFILE. Exiting."
    exit 1
fi

echo "Sourcing anaconda profile from $CONDA_PROFILE"
source $CONDA_PROFILE
echo "Sourced Conda.sh script. Now activating environment"

if ! conda_env_exists_name $ENV_NAME
then
    echo "Desired Conda environment $ENV_NAME does not exist. Aborting."
    exit 1
fi

echo "Now activating Conda environment: $ENV_NAME..."

conda activate $ENV_NAME

if [[ $CONDA_DEFAULT_ENV && $ENV_NAME = *$CONDA_DEFAULT_ENV ]]
then
    echo "Successfully activated environment at $ENV_NAME. Conda env is now $CONDA_DEFAULT_ENV"
else
    echo "Could not activate environment at $ENV_NAME; instead activated: $CONDA_DEFAULT_ENV. Aborting."
    exit 1
fi

python finetune_transformer.py $TRAIN_LANG $RUN_ID

echo "Done."
conda deactivate
