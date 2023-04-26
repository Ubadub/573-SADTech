#!/bin/sh

echo "Activating Environment"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /home2/taraw28/miniconda3/envs/SADTech

# echo "Creating Database"
# python3 src/run_preprocessing.py tam
# python3 src/run_preprocessing.py mal

echo "Resetting Output/Results Files"

> outputs/D2/mal/nb_output.txt
> outputs/D2/tam/nb_output.txt
> results/D2_scores.out

echo "Running Baseline: Naive Bayes Classifier"
python src/multinomial_nb_classifier.py src/config/nb_tam.yml
python src/multinomial_nb_classifier.py src/config/nb_mal.yml

echo "Running Finetuned Transformer LM Inference - TAMIL"
echo "### TAMIL ###" >> results/D2_scores.out
python src/transformer_inference.py tam >> results/D2_scores.out #outputs/D2/tam/transformers_output.txt

echo "Running Finetuned Transformer LM Inference - MALAYALAM"
echo "### MALAYALAM ###" >> results/D2_scores.out
python src/transformer_inference.py mal >> results/D2_scores.out #outputs/D2/mal/transformers_output.txt

echo "DONE"
