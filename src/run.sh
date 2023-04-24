#!/bin/sh

# ./run.sh config.yml

./preprocess.sh

./classifier.sh $1

./eval.sh $1