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

def classfiy(inX, dataSet, labels, k):
    m,n = np.shape(dataSet)
    diffMat = np.tile(inX, (m,1)) - dataSet
    sqDistSum = np.sum(diffMat ** 2, axis=1, dtype=np.float32)
    distances = sqDistSum ** 0.5
    sortedDistance = np.argsort(distances, axis=-1)
    classCount = {}
    for i in range(k):
      voteLable = labels[sortedDistance[i]]
      classCount[voteLable]= classCount.get(voteLable, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fb = open(filename)
    arrayOfLines = fb.readlines()
    numberOfLines= len(arrayOfLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabels = []
    for line in arrayOfLines:
        index = arrayOfLines.index(line)
        line = line.strip()
        returnMat[index:] = line.split('\t')[:3]
        classLabels.append(line.split('\t')[-1])
    return returnMat,classLabels

def autoNormal(dataSet):
    minVal = dataSet.min(0)
    maxVal = dataSet.max(0)
    ranges = maxVal - minVal
    m,n = np.shape(dataSet)
    normalDataSet = dataSet - np.tile(minVal, (m, 1))
    normalDataSet = normalDataSet/np.tile(ranges, (m,1))
    return normalDataSet, ranges, minVal


# returnMat,classLabels = file2matrix('datingTestSet2.txt')
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(returnMat[:,0], returnMat[:,1], 15.0*np.array(classLabels[:-10], dtype=np.float32), 15.0*np.array(classLabels, dtype=np.float32))
# plt.show()


def datingTest():
    testRating = 0.1
    datingDataSet, datingLabels = file2matrix('datingTestSet2.txt')
    normalMat, ranges, minVal = autoNormal(datingDataSet)
    m = np.shape(normalMat)[0]
    testNum =  int(testRating * m)
    errorCount = 0.0
    for i in range(testNum):
        testLabel = classfiy(normalMat[i,:], normalMat[testNum:m,:], datingLabels[testNum:m], 3)
        if testLabel != datingLabels[i]:
            errorCount += 1.0
        print('the classifier come back is %s, the real answer is %s') % (testLabel, datingLabels[i])
    print ('the total error rate is %f') %(errorCount/testNum)


datingTest()





