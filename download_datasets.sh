#!/bin/bash

mkdir datasets
wget -O datasets/triviaqa.tar.gz http://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz
wget -O datasets/sciQ.zip https://ai2-public-datasets.s3.amazonaws.com/sciq/SciQ.zip
git clone https://github.com/sylinrl/TruthfulQA.git
mv TruthfulQA/TruthfulQA.csv datasets/truthfulQA.csv

mkdir datasets/triviaqa
tar -xvf datasets/triviaqa.tar.gz -C datasets/triviaqa
unzip datasets/sciQ.zip -d datasets/sciq
rm datasets/sciQ.zip
rm datasets/triviaqa.tar.gz
rm -rf TruthfulQA