#!/bin/bash

#SBATCH --job-name=augment_573_SADTech
#SBATCH --mail-user=abhinavp@uw.edu

#SBATCH --account=stf
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=5:00:00 # Max runtime in DD-HH:MM:SS format.

#SBATCH --chdir=/mmfs1/gscratch/stf/abhinavp/573-SADTech/src/
#SBATCH --export=all
#SBATCH --output=/mmfs1/home/abhinavp/.logs/%x_%j.out # where STDOUT goes
#SBATCH --error=/mmfs1/home/abhinavp/.logs/%x_%j.err # where STDERR goes

./run_with_conda.hyak.sh "SADTech" 

# ./run_with_conda.hyak.sh "SADTech" python inference.py --config-name=inference lang=tam '+pipeline_path="../d4_models/primary/tam/"' '+outputs.sub_dir=primary' '+results.sub_dir=primary' 'hydra.verbose=True'
# ./run_with_conda.hyak.sh "SADTech" python inference.py --config-name=inference lang=mal '+pipeline_path="../d4_models/primary/mal/"' '+outputs.sub_dir=primary' '+results.sub_dir=primary' 'hydra.verbose=True'
# ./run_with_conda.hyak.sh "SADTech" python inference.py --config-name=inference lang=tam_mal_shuffled '+pipeline_path="../d4_models/adaptation/"' '+outputs.sub_dir=adaptation' '+results.sub_dir=adaptation' 'hydra.verbose=True'

# Ignored:
#SBATCH --output=/mmfs1/gscratch/stf/abhinavp/573-SADTech/slurm_output/D2/err.err # where STDOUT goes
#SBATCH --error=/mmfs1/gscratch/stf/abhinavp/573-SADTech/slurm_output/D2/out.out # where STDERR goes
