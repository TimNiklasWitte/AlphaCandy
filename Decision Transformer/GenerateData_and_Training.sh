#!/bin/bash

#set -x #echo on

#fieldSize=5
numCandys=5
trainSize=20000
numEpochs=10


for fieldSize in {5..8}
do
    #echo "fieldSize=$fieldSize numCandys=$numCandys trainSize=$trainSize"
    python3 GenerateTrainingData.py --fieldSize $fieldSize --numCandys $numCandys --trainSize $trainSize

    python3 Training.py --fieldSize $fieldSize --numCandys $numCandys --trainSize $trainSize --epochs $numEpochs

done
