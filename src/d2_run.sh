#!/bin/sh

echo "Activating Environment"
source /home2/taraw28/miniconda3/etc/profile.d/conda.sh
conda activate /home2/taraw28/miniconda3/envs/SADTech

echo "Creating Database"
python3 src/run_preprocessing.py tam
python3 src/run_preprocessing.py mal

echo "Resetting Output/Results Files"
> outputs/D2/mal/nb_output.txt
> outputs/D2/tam/nb_output.txt
> results/D2_scores.out

echo "Running Baseline: Naive Bayes Classifier"
python3 src/multinomial_nb_classifier.py src/config/nb_tam.yml
python3 src/multinomial_nb_classifier.py src/config/nb_mal.yml

# Run Finetine Transformer (FT)
# python src/finetune_transformer.py...

echo "DONE"