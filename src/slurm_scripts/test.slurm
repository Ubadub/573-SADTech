#!/bin/bash

#SBATCH --job-name=slurm_test
#SBATCH --mail-user=abhinavp@uw.edu
#SBATCH --mail-type=ALL

#SBATCH --account=stf
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=0:15:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/mmfs1/gscratch/stf/abhinavp/573-SADTech/src/
#SBATCH --export=all
#SBATCH --output=/mmfs1/home/abhinavp/.logs/%x_%j.out # where STDOUT goes
#SBATCH --error=/mmfs1/home/abhinavp/.logs/%x_%j.err # where STDERR goes

# Modules to use (optional).
# <e.g., module load apptainer>

srun echo "hello world"

# disregarded:
#SBATCH --open-mode=append # append or truncate
