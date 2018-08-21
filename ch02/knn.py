#-*- coding:utf-8 -*-
import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt

def CreateDataSet():
    group = np.array([[1.0, 1.1], [1.0,1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group,labels

# can optimize, change the distance method. to be continue

def knn(inX, dataSet, labels, k):
    m,n = np.shape(dataSet)
    diffMat = np.tile(inX, (m,1)) - dataSet
    sqDistSum = sum(diffMat ** 2, axis=1, dtype=np.float32)
    distances = sqDistSum ** 0.5
    sortedDistance = np.argsort(distances, axis=-1)
    classCount = {}
    for i in range(k):
      voteLable = labels[sortedDistance[i]]
      classCount[voteLable]= classCount.get(voteLable, 0) + 1
    sortedClassCount = sorted(classCount, key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(path):
    pass






