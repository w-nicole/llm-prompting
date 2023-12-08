#!/bin/bash

mkdir datasets
mkdir datasets/raw
wget -O datasets/raw/triviaqa.tar.gz http://nlp.cs.washington.edu/triviaqa/data/triviaqa-unfiltered.tar.gz
wget -O datasets/raw/sciQ.zip https://ai2-public-datasets.s3.amazonaws.com/sciq/SciQ.zip
git clone https://github.com/sylinrl/TruthfulQA.git
mv TruthfulQA/TruthfulQA.csv datasets/raw/truthfulQA.csv

mkdir datasets/raw/triviaqa
tar -xvf datasets/raw/triviaqa.tar.gz -C datasets/raw/triviaqa
unzip datasets/raw/sciQ.zip -d datasets/raw/sciq
rm datasets/raw/sciQ.zip
rm datasets/raw/triviaqa.tar.gz
rm -rf TruthfulQA