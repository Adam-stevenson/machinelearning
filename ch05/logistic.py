#-*- coding:utf-8 -*-
from numpy import *

def loadDataSet(filename):
    dataMat = []
    labels = []
    fb = open(filename, 'r')
    for line in fb.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labels.append(lineArr[-1])
    return dataMat,labels

def sigmoid(inX):
    return 1/(1+exp(-inX))

def gradAscent(dataMatIn, labels):
    dataMat = mat(dataMatIn)
    labelMat = mat(labels).transpose()
    m,n = shape(dataMat)
    alpha = 0.01
    maxCycle = 500
    weights = ones((n, 1))
    for i in range(maxCycle):
        h = sigmoid(dataMat*weights)
        error = labelMat-h
        weights = weights + alpha*dataMat.transpose()*error
    return weights

def randomGradAscent(dataMatIn, labels):
    m,n = shape(dataMatIn)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(dataMatIn[i]*weights)
        error = labels[i] - h
        weights = weights + alpha*error*dataMatIn[i]
    return weights

def stocGradAscent(dataMat, labels, numIter=50):
    m, n = shape(dataMat)
    weight = ones(n)
    for j in range(numIter):
        dataInx = range(m)
        for i in range(m):
            alpha = 4/(i+j+1.0) + 0.01
            randInx = int(random.uniform(0, len(dataInx)))
            h = sigmoid(sum(dataMat[randInx]*labels))
            error =  labels[randInx] - h
            weight = weight + alpha*error*dataMat[randInx]
            del(dataMat[randInx])
    return weight

def classify(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob>0.5:
        return 1
    else:
        return 0

