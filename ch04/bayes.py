#-*- coding:utf-8 -*-
from numpy import *
def loadDataSet():
    dataSet = [['my','dog','has','flea','problems','help','please'],
               ['maybe','not','take','him','to','dog','park','stupid'],
               ['my','dalmation','is','so','cute','I','love','him'],
                ['stop','posting','stupid','worthless','garbage' ],
                ['mr','licks','ate','my','steak','how','to','stop','him'],
                ['quit', 'buying', 'worthless', 'dog','food','stupid']]
    labels = [0,1,0,1,0,1]
    return dataSet,labels

def createVocabList(dataSet):
    vocaSet = set({})
    for data in dataSet:
        vocaSet = vocaSet | set(data)
    return list(vocaSet)

def setOfwords2Vec(vocaList, input):
    returnVec = [0]*len(vocaList)
    for word in input:
        if word in vocaList:
            # returnVec[vocaList.index(voca)] = 1
            returnVec[vocaList.index(word)] += 1
    return returnVec

def traiN0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numberWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # p0Num = zeros(numberWords)
    # p1Num = zeros(numberWords)
    p0Num = ones(numberWords)
    p1Num = ones(numberWords)
    # p0Denom = 0.0
    # p1Denom = 0.0
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # p1Vec = p1Num / p1Denom
    # p0Vec = p0Num / p0Denom
    p1Vec = log(p1Num / p1Denom)
    p0Vec = log(p0Num / p0Denom)
    return p0Vec,p1Vec,pAbusive

def classify0(vec2Classify, p0Vec, p1Vec, pAbusive):
    p0 = sum(vec2Classify*p0Vec) + log(1-pAbusive)
    p1 = sum(vec2Classify*p1Vec + log(pAbusive))
    if p1>p0:
        return p1
    else: return p0
# dataSet,labels = loadDataSet()
# vocaList = createVocabList(dataSet)
# trainMat = []
# for data in dataSet:
#     vec =  setOfwords2Vec(vocaList, data)
#     trainMat.append(vec)
trainMat = [[0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]]
trainCategory = [0,1,0,1,0,1]
print traiN0(trainMat, trainCategory)
