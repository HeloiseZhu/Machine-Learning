#!/usr/bin/env python
# -*- coding:utf-8 -*-

import sys
from math import *
import numpy as np

def mylog(x):
    if x == 0:
        return 0
    else:
        return log2(x)

if __name__ == '__main__':
    input_file = sys.argv[1]
    x1 = []
    x2 = []
    y = []
    T1 = []
    T2 = []
    data = []
    ent = []

    #读数据
    with open(input_file, 'r') as fp:
        x1 = fp.readline().strip().split(' ')
        x2 = fp.readline().strip().split(' ')
        y = fp.readline().strip().split(' ')

    for i in range(len(x1)):
        temp = [int(x1[i]), int(x2[i]), int(y[i])]
        data.append(temp)
    
    data = np.array(data)
    data = data[data[:, 0].argsort()]
    n = len(x1)
    print(data)

    #计算Ent(D)
    pNum = 0
    for d in data:
        if d[2] == 1:
            pNum += 1
    p0 = pNum / n
    p1 = (n - pNum) / n
    Ent = -p0 * log2(p0) - p1 * log2(p1)
    print("Ent(D): ", Ent)
    
    for i in range(n - 1):
        cVal = (data[i][0] + data[i+1][0]) / 2.0
        T1.append(cVal)
    print("T1: ",T1)

    for i in range(n-1):
        w1 = (i + 1) / n
        w2 = (n - i - 1) / n
        pNum1 = 0
        pNum2 = 0
        for j in range(i + 1):
            if data[j][2]==1:
                pNum1 += 1
        p01 = (i+1-pNum1) / (i+1)
        p11 = pNum1 / (i+1)
        for j in range(i + 1, n):
            if data[j][2]==1:
                pNum2 += 1
        p02 = (n - i - 1-pNum2) / (n-i-1)
        p12 = pNum2 / (n - i - 1)
        #print(w1, p01, p11)
        #print(w2, p02, p12)
        temp = Ent + w1 * (p01 * mylog(p01) + p11 * mylog(p11)) + w2 * (p02 * mylog(p02) + p12 * mylog(p12))
        ent.append(temp)
        print(temp)
        