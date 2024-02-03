#!/bin/bash

# Parameters
#SBATCH --account=stf
#SBATCH --array=0-39%40
#SBATCH --error=/mmfs1/gscratch/stf/abhinavp/573-SADTech/src/testing_job_arrays/logs/%A_%a_%j.err
#SBATCH --gres=gpu:rtx6k:1
#SBATCH --job-name=__main__
#SBATCH --mem=36GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/mmfs1/gscratch/stf/abhinavp/573-SADTech/src/testing_job_arrays/logs/%A_%a_%j.out
#SBATCH --partition=ckpt
#SBATCH --signal=USR2@120
#SBATCH --time=2880

# command
srun /usr/bin/zsh my_script.sh %A %a %j
# srun --output /mmfs1/gscratch/stf/abhinavp/573-SADTech/src/testing_job_arrays/logs/%A_%a/%A_%a_%t_log.out --error /mmfs1/gscratch/stf/abhinavp/573-SADTech/src/testing_job_arrays/logs/%A_%a/%A_%a_%t_log.err /usr/bin/zsh my_script.sh %A %a %j
# srun --unbuffered --output /mmfs1/gscratch/stf/abhinavp/573-SADTech/src/testing_job_arrays/logs/%A_%a/%A_%a_%t_log.out --error /mmfs1/gscratch/stf/abhinavp/573-SADTech/src/testing_job_arrays/logs/%A_%a/%A_%a_%t_log.err /usr/bin/zsh my_script.sh %A %a %j

#SBATCH --wckey=submitit
