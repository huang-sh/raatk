#!/usr/bin/env bash

set -uex

# view reduced amino acid alphabets
raatk view -t 1 2 -s 2 4 6 9

# reduce amino acid sequence
raatk reduce positive.txt negative.txt -t 1-8 -s 2-19 -o pos neg

# extract sequence feature
raatk extract pos neg -k 3 -d -o k3 -m

# evaluation
raatk eval k3 -d -o k3-eval -cv 5 -clf svm -c 2 -g 0.5 -p 3
raatk eval k3/type2/10-ARNCQHIFPW.csv -cv -1 -c 2 -g 0.5 -o k3-t2s10.txt

# result visualization
raatk plot k3-eval.json -o k3-p

# ROC evaluation
raatk roc k3/type2/10-ARNCQHIFPW.csv -clf svm -cv 5 -c 2 -g 0.5 -o roc

# feature selection
raatk ifs k3/type2/10-ARNCQHIFPW.csv -s 2 -clf svm -cv 5 -c 2 -g 0.5 -o ifs

# feature visulization
raatk fv ifs_56-best.csv -o ifs-fv



# model training and prediction

cluster='AGST-RK-ND-C-QE-H-ILMV-FY-P-W'
raa='ARNCQHIFPW'
feature='ifs_56-best.csv'
idx='ifs-56-idx.csv'
model='svm.model'
testSeq='test_seq.txt'
rSeq='reduce_seq.txt'
testFeature='test_feature.csv'
testResult='test_result.csv'

# train SVM classifier
raatk train $feature -clf svm -c 2 -g 0.5 -o $model -prob
# reduce test sequence
raatk reduce $testSeq -c $cluster -o $rSeq
# extract sequence feature
raatk extract $rSeq -raa $raa -k 3 -idx $idx -o $testFeature --label-f
# predcit test sequence
raatk predict $testFeature -m $model -o $testResult

