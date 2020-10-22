from numpy import *
from pandas import *


def sigmoid(x):
    res = 1.0 / (1.0 + exp(-x))
    return res

class LRClassifier:
    trainSet = []
    dataMatrix = []
    label = []
    weights = []

    testSet = []
    testDataMatrix = []
    testResult = []


    def getDataSet(self):
        # train set
        # 分离示例（dataMatrix）和标记（label）
        self.trainSet = read_csv("train_set.csv").values
        self.label = self.trainSet[:, 16].reshape((-1,1))   # 列向量
        self.dataMatrix = delete(self.trainSet, 16, axis=1)
        self.dataMatrix = self.normalize(self.dataMatrix)
        # test set
        self.testSet = read_csv("test_set.csv").values
        self.testResult = self.testSet[:,16]                # 行向量
        self.testDataMatrix = delete(self.testSet, 16, axis=1)
        self.testDataMatrix = self.normalize(self.testDataMatrix)


    def OvR(self):
        X = column_stack((ones(self.dataMatrix.shape[0]), self.dataMatrix))
        for i in range(1, 27):
            print("第", i, "个OvR分类器")
            ovrLabel = zeros(self.label.shape)
            # 1种为正例，25种为反例
            for j in range(self.label.shape[0]):
                if(self.label[j][0] == i):
                    ovrLabel[j][0] = 1
            ret = self.newtonMethod(mat(X), mat(ovrLabel))
            self.weights.append(ret)
            print(ret)


    def newtonMethod(self, X, y, iteration=5):
        m, n = X.shape              # m个示例，每个示例n-1=16个特征
        beta = zeros((n, 1))     # (w;b)
        # 迭代
        for cnt in range(iteration):
            print("第",cnt+1,"次迭代")
            # l_first = np.zeros((n, 1))    # 一阶导数 / 梯度
            # l_second = np.zeros((n, n))   # 二阶导数 / Hessian矩阵
            h = sigmoid((X * beta))         # p_1
            l_first =  X.T * (h - y) / double(m)
            l_second = (X.T * diag(multiply(h, (1 - h)).getA1()) * X) / double(m)
            beta = beta - l_second.I * l_first
        return beta
        

    def classify(self):
        TP = zeros(26)
        FP = zeros(26)
        TN = zeros(26)
        FN = zeros(26)
        P = zeros(26)
        R = zeros(26)
        F1 = zeros(26)
        accurateNum = 0
        X = column_stack((ones(self.testDataMatrix.shape[0]), self.testDataMatrix))
        testNum = X.shape[0]

        for i in range(testNum):
            maxProb = 0
            letterClass = 0
            for j in range(26): 
                predictVal = sigmoid(double(X[i].reshape((1,-1)) * self.weights[j]))
                if predictVal > maxProb:
                    maxProb = predictVal
                    letterClass = j + 1

            if letterClass == self.testResult[i]:
                accurateNum += 1
                # 以标记所在类为正例的分类器中，TP加1
                TP[self.testResult[i] - 1] += 1
                # 以标记所在类为反例的分类器中，TN加1
                TN = TN + ones(26)
                TN[self.testResult[i] - 1] -= 1
            else:
                FN[self.testResult[i] - 1] += 1
                FP[letterClass - 1] += 1
                for k in range(26):
                    if k != self.testResult[i] - 1 and k != letterClass - 1:
                        TN[k] += 1

        # 性能指标
        for i in range(26):
            P[i] = TP[i] / (TP[i] + FP[i])
            R[i] = TP[i] / (TP[i] + FN[i])
            F1[i] = 2 * P[i] * R[i] / (P[i] + R[i])
        tTP = sum(TP)
        tFP = sum(FP)
        tFN = sum(FN)
        microP = tTP / (tTP + tFP)
        microR = tTP / (tTP + tFN)
        microF1 = 2 * microP * microR / (microP + microR)
        macroP = mean(P)
        macroR = mean(R)
        macroF1 = mean(F1)

        print("\nTesting Performance")
        print("Accuracy: ", ("%.2f" % (accurateNum / testNum * 100)) , "%")
        print("micro Precision: ", ("%.2f" % (microP * 100)) , "%")
        print("micro Recall: ", ("%.2f" % (microR * 100)) , "%")
        print("micro F1: ", ("%.2f" % (microF1 * 100)) , "%")
        print("macro Precision: ", ("%.2f" % (macroP * 100)) , "%")
        print("macro Recall: ", ("%.2f" % (macroR * 100)) , "%")
        print("macro F1: ", ("%.2f" % (macroF1 * 100)) , "%")
        
    
    def normalize(self, testDataMat):
        maxVal = testDataMat.max(0)
        minVal = testDataMat.min(0)
        m, n = testDataMat.shape
        tmpMat = double(testDataMat)
        for j in range(n):
            tmp = double(maxVal[j] - minVal[j])
            for i in range(m):
                tmpMat[i][j] = (tmpMat[i][j] - minVal[j]) / tmp
        return tmpMat
                    


if __name__ == '__main__':
    m = LRClassifier()
    m.getDataSet()
    m.OvR()
    m.classify()