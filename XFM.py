# -*- coding: utf-8 -*-

from __future__ import division
from math import exp
from numpy import *
from pandas import concat
from random import normalvariate  # 正态分布
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse


def preprocessData(data,category_features,numeric_features,label):

    lenTrain = len(data[0])
    xgbFeatures = XGB(data[0], data[1], numeric_features, label)
    data = concat(data)
    label = data[label].map(lambda x: 1 if x == 1 else -1)  # 取标签并转化为 +1，-1

    xgbFeatures = OneHotEncoder().fit_transform(xgbFeatures)
    catfeatures = OneHotEncoder().fit_transform(data[category_features].values)

    Data = sparse.hstack([xgbFeatures, catfeatures]).toarray()

    return Data[:lenTrain,:],Data[lenTrain:,:],label[:lenTrain],label[lenTrain:]

def sigmoid(inx):
    #return 1. / (1. + exp(-max(min(inx, 15.), -15.)))
    return 1.0 / (1 + exp(-inx))

def XGB(train,test,numeric_features,label):
    n_feats = len(numeric_features)
    xgboost = xgb.XGBClassifier(nthread=4, learning_rate=0.5,
                                n_estimators=n_feats, max_depth=6,
                                gamma=0.05, subsample=0.5,
                                colsample_bytree=1)
    print("training: ", numeric_features)
    xgboost.fit(train[numeric_features], train[label])

    train_leaves = xgboost.apply(train[numeric_features])
    test_leaves = xgboost.apply(test[numeric_features])

    xgbFeatures = concatenate((train_leaves, test_leaves), axis=0)

    return xgbFeatures


def SGD_FM(dataMatrix, classLabels, k, iter, early_stop):
    '''
    :param dataMatrix:  特征矩阵
    :param classLabels: 类别矩阵
    :param k:           辅助向量的大小
    :param iter:        迭代次数
    :return:
    '''
    # dataMatrix用的是mat, classLabels是列表
    m, n = shape(dataMatrix)   #矩阵的行列数，即样本数和特征数
    alpha = 0.01
    # 初始化参数
    # w = random.randn(n, 1)#其中n是特征的个数
    w = zeros((n, 1))      #一阶特征的系数
    w_0 = 0.
    v = normalvariate(0, 0.2) * ones((n, k))
    V_List = [];W0_List = [];W_List = []
    min_loss = m
    best_round = 0
    stop_rate = 0

    for it in range(iter):
        cost = 0
        for x in range(m):  # 随机优化，每次只使用一个样本
            # 二阶项的计算
            inter_1 = dataMatrix[x] * v
            inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)  #二阶交叉项的计算
            interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.       #二阶交叉项计算完成

            p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出，即FM的全部项之和
            loss = 1-sigmoid(classLabels[x] * p[0, 0])    #计算损失
            cost += loss

            w_0 = w_0 +alpha * loss * classLabels[x]

            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] +alpha * loss * classLabels[x] * dataMatrix[x, i]
                    for j in range(k):
                        v[i, j] = v[i, j]+ alpha * loss * classLabels[x] * (
                        dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])

        W0_List.append(v);W_List.append(v);V_List.append(v)
        print("第{}次迭代后的平均损失为: {}".format(it, cost/m))
        if cost < min_loss:
            min_loss = cost
            best_round = it
            stop_rate = 0
        else:
            stop_rate += 1
        if stop_rate >= early_stop:
            break

    print("最优迭代次数 = {},最优的平均Loss = {}".format(best_round,min_loss/m))

    return W0_List[best_round],W_List[best_round],V_List[best_round] #w_0, w, v

def predictXFM(dataMatrix, w_0, w, v):
    m, n = shape(dataMatrix)
    allItem = 0
    result = []
    for x in range(m):   #计算每一个样本的误差
        allItem += 1
        inter_1 = dataMatrix[x] * v
        inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)
        interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.
        p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出

        pre = sigmoid(p[0, 0])
        result.append(pre)

    return result



