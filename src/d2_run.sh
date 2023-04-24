#!/bin/sh

source /home2/taraw28/miniconda3/etc/profile.d/conda.sh
conda activate /home2/taraw28/573-SADTech/environment.yml


# Create Database
#   Do we need to call this or are we submitting a file that already contains the database?
python src/preprocessing/dataset_creation.py

# Run Baseline (Naiive Bayes Classifier)
python src/multinomial_nb_classifier.py src/config/config1.yml

# Do Baseline Eval

# Run Finetine Transformer (FT)
# python src/finetune_transformer.py...

# Do FT Eval