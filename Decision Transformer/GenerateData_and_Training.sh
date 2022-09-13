#!/bin/bash

#set -x #echo on

trainSize=50000
numEpochs=20

for numCandys in {4..6}
do

    for fieldSize in {5..8}
    do

        python3 GenerateTrainingData.py --fieldSize $fieldSize --numCandys $numCandys --trainSize $trainSize
        python3 Training.py --fieldSize $fieldSize --numCandys $numCandys --trainSize $trainSize --epochs $numEpochs

    done

done