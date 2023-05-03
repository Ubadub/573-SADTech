#!/bin/bash

#SBATCH --job-name=SADTech_mypy
#SBATCH --mail-user=abhinavp@uw.edu

#SBATCH --account=stf
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=0:20:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --export=all
#SBATCH --output=/mmfs1/home/abhinavp/.logs/%x_%j.out # where STDOUT goes
#SBATCH --error=/mmfs1/home/abhinavp/.logs/%x_%j.err # where STDERR goes

# Modules to use (optional).
# <e.g., module load apptainer>

./run_with_conda.hyak.sh SADTech mypy finetune_transformer.py --check-untyped-defs


# disregarded:
#SBATCH --mail-type=END
#SBATCH --open-mode=append # append or truncate
