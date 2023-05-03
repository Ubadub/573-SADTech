#!/bin/bash

#SBATCH --job-name=D2_573_SADTech
#SBATCH --mail-user=abhinavp@uw.edu

#SBATCH --account=stf
#SBATCH --partition=compute-hugemem
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=1:30:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/mmfs1/gscratch/stf/abhinavp/573-SADTech/src/
#SBATCH --export=all
#SBATCH --output=/mmfs1/gscratch/stf/abhinavp/573-SADTech/slurm_output/D2/err.err # where STDOUT goes
#SBATCH --error=/mmfs1/gscratch/stf/abhinavp/573-SADTech/slurm_output/D2/out.out # where STDERR goes

./run_with_conda.hyak.sh "SADTech" ./d2_run.sh "yes"

# Ignored:
#SBATCH --output=/mmfs1/home/abhinavp/.logs/%x_%j.out # where STDOUT goes
#SBATCH --error=/mmfs1/home/abhinavp/.logs/%x_%j.err # where STDERR goes
