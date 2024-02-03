#!/bin/bash

# set -euxo pipefail # automatically exit if any program errors, prohibit undefined variables
# set -euo pipefail # automatically exit if any program errors, prohibit undefined variables
set -eo pipefail # automatically exit if any program errors, prevent pipeline errors from being masked

# This script sets up the CPU conda environment on a CPU node.

# if you install anaconda in a different directory, change the following line to
# CONDA_PROFILE=path_to_anaconda3/anaconda3/etc/profile.d/conda.sh
CONDA_PROFILE=$(realpath /mmfs1/gscratch/stf/abhinavp/miniconda3/etc/profile.d/conda.sh)

EXTRA_CHANNEL_FLAGS="-c conda-forge"

YML_FILE=$(realpath "${1:-/projects/assigned/lm-inductive/corpus-filtering/environment.yml}")
# YML_FILE=$(realpath "${1:-../environment.yml}")
# ENV_PREFIX=$(realpath "${2:-/projects/assigned/lm-inductive/envs/corpus_filter_env}")
# ENV_PREFIX=$(realpath "${2:-../../envs/corpus_filter_env}")
ENV_NAME="${2:-corpus_filter_env}"

conda_env_exists_prefix(){
    conda env list | grep -E "^.*\s+${@}$" >/dev/null 2>/dev/null
}

conda_env_exists_name(){
    conda env list | grep -E "^${@}\s" >/dev/null 2>/dev/null
}

# echo "Using prefix: $ENV_PREFIX"
echo "Using environment name: $ENV_NAME"
echo "Using YML file: $YML_FILE"

if [[ ! -f $CONDA_PROFILE ]] 
then
    echo "Conda profile not found at $CONDA_PROFILE. Exiting."
    exit 1
fi

if [[ ! -f $YML_FILE ]] 
then
    echo "Environment file not found at $YML_FILE. Exiting."
    exit 1
fi

# echo "Sourcing anaconda profile from $CONDA_PROFILE"
# source $CONDA_PROFILE
# echo "Sourced Conda.sh script. Now creating environment"

echo "Sourcing anaconda profile from $CONDA_PROFILE"
source $CONDA_PROFILE
echo "Sourced Conda.sh script. Now activating base environment"

conda activate base
if [[ $CONDA_DEFAULT_ENV && "base" == $CONDA_DEFAULT_ENV ]]
then
    echo "Successfully activated base environment."
else
    echo "Could not activate base environment. Aborting."
    exit 1
fi

if ! conda_env_exists_name $ENV_NAME
then
    echo "Conda environment does not exist. Creating environment at $ENV_NAME."
    conda create -n $ENV_NAME --strict-channel-priority --yes 
else
    echo "Conda environment exists. Proceeding with update."
fi

if ! conda_env_exists_name $ENV_NAME # check it was created successfully above
then
    echo "Failed to create Conda environment. Aborting."
    exit 1
fi

echo "Now activating Conda environment: $ENV_NAME..."

# deactivate base environment
conda deactivate

conda activate $ENV_NAME

if [[ $CONDA_DEFAULT_ENV && $ENV_NAME = *$CONDA_DEFAULT_ENV ]]
then
    echo "Successfully activated environment at $ENV_NAME. Conda env is now $CONDA_DEFAULT_ENV"
else
    echo "Could not activate environment at $ENV_NAME; instead activated: $CONDA_DEFAULT_ENV. Aborting."
    exit 1
fi

if [[ ! $(conda config --env --show channel_priority | grep "strict") ]]
then
    echo "Channel priority not strict; aborting."
    exit 1
fi

# temporary
conda config --env --set always_yes true

echo "Updating environment to use specifications in $YML_FILE"
conda env update -f $YML_FILE -n $ENV_NAME --prune --json

# reset
conda config --env --remove-key always_yes
# conda env update -f $YML_FILE -p $ENV_NAME --prune
# conda env update -f environment.yml -p /projects/assigned/lm-inductive/envs/corpus_filter_env --prune

echo "Done updating environment. You may want to double check the desired packages are available."
conda deactivate
