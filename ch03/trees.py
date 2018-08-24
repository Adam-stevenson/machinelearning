#-*- coding:utf-8 -*-

from numpy import *
# from math import log
import numpy
import operator

def calShannonEnt(dataSet):
    m,n = shape(dataSet)
    featLables = {}
    for featVec in dataSet:
        featLabel = featVec[-1]
        featLables[featLabel] = featLables.get(featLabel, 0) + 1
    shannonEnt = 0.0
    for key in featLables.keys():
        prob = float(featLables[key])/m
        shannonEnt -= prob * numpy.math.log(prob, 2)
    return shannonEnt

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            sonfeatVec = featVec[:axis]
            sonfeatVec.extend(featVec[axis+1:])
            retDataSet.append(sonfeatVec)
    return retDataSet

def chooseBestSplit(dataSet):
    numFeatures = shape(dataSet)[1] - 1
    baseEntropy = calShannonEnt(dataSet)
    bestFeatureIndex = -1
    bestInfoGain = 0.0
    for i in range(numFeatures):
        tmp = [example[i] for example in dataSet]
        uniqueVal = set(tmp)
        newEntroy = 0.0
        for val in uniqueVal:
            childDataSet = splitDataSet(dataSet, i, val)
            prob = float(len(childDataSet))/len(dataSet)
            newEntroy += prob * calShannonEnt(childDataSet)
        infoGain = baseEntropy  - newEntroy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeatureIndex = i
    return bestFeatureIndex

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        classCount[vote] = classCount.get(vote, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    uniqClassList = set(classList)
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    if len(uniqClassList) == 1:
        return classList[0]
    bestFeat = chooseBestSplit(dataSet)
    bestLable = labels[bestFeat]
    trees = {bestLable:{}}
    del(labels[bestFeat])
    uniVals = set([example[bestFeat] for example in dataSet])
    for val in uniVals:
        subLabels = labels[:]
        trees[bestLable][val] = createTree(splitDataSet(dataSet, bestFeat, val), subLabels)
    return trees


def createDataSet():
    dataSet=[
    [1,1,'yes'],
    [1,1,'yes'],
    [1,0,'no'],
    [0,1,'no'],
    [0,1,'no']]
    labels = ['nosurfacing','flippers']
    return dataSet,labels

# dataSet,labels = createDataSet()
# print splitDataSet(dataSet, 1, 1)
# print calShannonEnt(dataSet)
# print chooseBestSplit(dataSet)
# print createTree(dataSet, labels)

####### above ID3
import matplotlib.pyplot as plt
decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt , parentPt , nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', xytext=centerPt,textcoords="axes fraction",
                            va='center', ha='center', bbox=nodeType,arrowprops=arrow_args)

# def createPlot():
#     fig = plt.figure(1, facecolor='white')
#     fig.clf()
#     createPlot.ax1 = plt.subplot(111, frameon = False)
#     plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
#     plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
#     plt.show()
# createPlot()

def getNumLeafs(myTree):
    leafNum = 0
    firstLeaf = myTree.keys()[0]
    secondDict = myTree[firstLeaf]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            leafNum += getNumLeafs(secondDict[key])
        else:leafNum += 1
    return leafNum

def getTreeDepth(myTree):
    maxDepth = 0
    firstLeaf = myTree.keys()[0]
    secondDict = myTree[firstLeaf]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth>maxDepth: maxDepth = thisDepth
    return maxDepth

def plotMidText(centerPt, parentPt, txtString):
    xMid= (parentPt[0] - centerPt[0])/2.0 + centerPt[0]
    yMid= (parentPt[1] - centerPt[1])/2.0 + centerPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):#if the first key tells you what feat was split on
    numLeafs = getNumLeafs(myTree)  #this determines the x width of this tree
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]     #the text label for this node should be this
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
            plotTree(secondDict[key],cntrPt,str(key))        #recursion
        else:   #it's a leaf node print the leaf node
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD

def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()
# myTree = {'nosurfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
# myTree['nosurfacing'][3] = 'mabe'
# createPlot(myTree)

########################

def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    index = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[index] == key:
            if type(secondDict[key]).__name__=='dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

# myTree = {'nosurfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
# labels = ['nosurfacing','flippers']
# print classify(myTree, labels, [1, 1])

########C4.5

def calcuSplitInfo(featList):
    featNum = len(featList)
    featSet = set(featList)
    probList = [featList.count(item)/float(featNum) for item in featSet]
    infoList = [item*numpy.math.log(item, 2) for item in probList]
    shannonEnt = -sum(infoList)
    return shannonEnt,featSet

def chooseBestSplit1(dataSet):
    numFeatures = shape(dataSet)[1] - 1
    baseShannonEnt = calShannonEnt(dataSet)

    featValsSetList = []
    shannonEntList = []
    conditionEnt = []
    for i in range(numFeatures):
        featValsList = [example[i] for example in dataSet]
        newShannonEnt,featValsSet = calcuSplitInfo(featValsList)
        featValsSetList.append(featValsSet)
        shannonEntList.append(newShannonEnt)
        resultGain = 0.0
        for val in featValsSet:
                subDataSet = splitDataSet(dataSet, i, val)
                appearNum = len(subDataSet)
                subShannonEnt = calShannonEnt(subDataSet)
                prob = appearNum/len(dataSet)
                resultGain += prob * subShannonEnt
        conditionEnt.append(resultGain)
    infoGainArray = baseShannonEnt*ones(numFeatures) - array(conditionEnt)
    infoGainRatio = infoGainArray/array(shannonEntList)
    bestFeatIndex = argsort(-infoGainRatio)[0]
    return bestFeatIndex,featValsSetList[bestFeatIndex]


def chooseBestSplit2(dataSet):
    numFeatures = shape(dataSet)[1] - 1
    baseShannonEnt = calShannonEnt(dataSet)
    bestFeatIndex = -1
    bestGainRatio = 0.0
    bestFeatSet = []
    for i in range(numFeatures):
        featValsList = [example[i] for example in dataSet]
        newSplitInfo,featValsSet  = calcuSplitInfo(featValsList)
        newShannonEnt = 0.0
        for val in featValsSet:
            subDataSet = splitDataSet(dataSet, i, val)
            prob = featValsList.count(val)/float(len(dataSet))
            newShannonEnt += prob*calShannonEnt(subDataSet)
        infoGain = baseShannonEnt - newShannonEnt
        infoGainRatio = infoGain/newSplitInfo
        if infoGainRatio > bestGainRatio:
            bestFeatIndex = i
            bestFeatSet = featValsSet
    return bestFeatIndex,bestFeatSet

def buildTreeC45(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    uniqClassList = set(classList)
    if len(uniqClassList) == 1:
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeatIndex,bestFeatSet = chooseBestSplit2(dataSet)
    bestFeatVal = labels[bestFeatIndex]
    trees = {bestFeatVal:{}}
    del (labels[bestFeatIndex])
    for val in bestFeatSet:
        subLabels = labels[:]
        subDataSet = splitDataSet(dataSet, bestFeatIndex, val)
        trees[bestFeatVal][val] = buildTreeC45(subDataSet, subLabels)
    return trees


dataSet,labels = createDataSet()
print buildTreeC45(dataSet, labels)
# print chooseBestSplit(dataSet)
# print chooseBestSplit1(dataSet)
# print chooseBestSplit2(dataSet)