import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
import matplotlib.pyplot as plt 


def load_data():
    train_raw = pd.read_csv('adult.data', header=None, sep = ', ', engine='python').values
    test_raw = pd.read_csv('adult.test', header=None, skiprows=1, sep=', ', engine='python').values
    return train_raw, test_raw


def preprocessing(train_raw, test_raw):
    #age
    workclass = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                 'Local-gov','State-gov', 'Without-pay', 'Never-worked']
    #fnlwgt: continuous.
    education = ['Bachelors', 'Some-college', '11th', 'HS-grad',
                 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th',
                 '7th-8th', '12th', 'Masters', '1st-4th',
                 '10th', 'Doctorate', '5th-6th', 'Preschool']
    #education-num: continuous.
    marital_status = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
                      'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
    occupation = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales',
                  'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                  'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
                  'Priv-house-serv', 'Protective-serv', 'Armed-Forces']
    relationship = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
    race = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
    sex = ['Female', 'Male']
    #capital-gain: continuous.
    #capital-loss: continuous.
    #hours-per-week: continuous.
    native_country = ['United-States', 'Cambodia', 'England', 'Puerto-Rico',
                      'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India',
                      'Japan', 'Greece', 'South', 'China',
                      'Cuba', 'Iran', 'Honduras', 'Philippines',
                      'Italy', 'Poland', 'Jamaica', 'Vietnam',
                      'Mexico', 'Portugal', 'Ireland', 'France',
                      'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan',
                      'Haiti', 'Columbia', 'Hungary', 'Guatemala',
                      'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia',
                      'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands']
    salary = ['<=50K', '>50K']
    
    # 处理缺失值
    imp = SimpleImputer(missing_values='?', strategy="most_frequent")
    train_set = imp.fit_transform(train_raw)
    test_set = imp.fit_transform(test_raw)

    # 数据集编码
    le = LabelEncoder()
    for i in range(train_set.shape[1]):
        if isinstance(train_set[0][i], str):
            if i == 1:
                le = le.fit(workclass)
            elif i == 3:
                le = le.fit(education)
            elif i == 5:
                le = le.fit(marital_status)
            elif i == 6:
                le = le.fit(occupation)
            elif i == 7:
                le = le.fit(relationship)
            elif i == 8:
                le = le.fit(race)
            elif i == 9:
                le = le.fit(sex)
            elif i == 13:
                le = le.fit(native_country)
            elif i == 14:
                le = le.fit(salary)
            train_set[:, i] = le.transform(train_set[:, i])
            test_set[:, i] = le.transform(test_set[:, i])
    
    # 分离属性和标记
    train_y = train_set[:, 14]      # 一维数组
    train_x = np.delete(train_set, 14, axis=1)
    test_y = test_set[:, 14]
    test_x = np.delete(test_set, 14, axis=1)
    train_y = 2 * (train_y - 0.5)
    test_y = 2 * (test_y - 0.5)

    return train_x, train_y, test_x, test_y


def sign(pred):
    result = []
    for i in range(pred.shape[0]):
        if pred[i] >= 0:
            result.append(1)
        else:
            result.append(-1)
    result = np.array(result)
    return result


def adaboost(train_x, train_y, test_x, test_y, T):
    m1 = train_x.shape[0]
    m2 = test_x.shape[0]
    w = np.ones(m1) / m1    # 样本权重
    a = np.zeros(T)         # 基学习器权重
    result = np.zeros(m2)
      
    for i in range(T):
        # 训练第i个基学习器
        dt = DecisionTreeClassifier(min_samples_split=20, random_state=1)#, max_depth=20
        dt.fit(train_x, train_y.astype('int'), sample_weight=w)
        pred = dt.predict(train_x)
        # 计算错误率和基学习器权重
        cmp = np.array(list(map(int, (pred != train_y))))
        err = np.dot(w, cmp)
        #print('error rate: ', err)
        if err > 0.5:
            break
        elif err == 0:
            a[i] = 1
        else:
            a[i] = 0.5 * np.log((1 - err) / err)
        # 更新样本权重
        w = w * np.exp((-a[i] * pred * train_y).astype('float'))
        w = w / np.sum(w)
        # 累加第i个基学习器的预测值
        result += (a[i] * dt.predict(test_x))
    return sign(result)


def cross_validation(train_x, train_y, test_x, test_y, max_learner_num):
    kf = KFold(n_splits=5)#, shuffle=True
    learner_num = 0
    max_auc = 0

    ###
    draw_y = []
    draw_x = list(range(1, max_learner_num + 1))
    ###

    for i in range(1, max_learner_num + 1):
        auc = 0
        for train_index, test_index in kf.split(train_x):
            pred = adaboost(train_x[train_index], train_y[train_index], train_x[test_index], train_y[test_index], i)
            fpr, tpr, thresholds = metrics.roc_curve(train_y[test_index].astype('int'), pred)
            auc += metrics.auc(fpr, tpr)
            #print('AUC: ', metrics.auc(fpr, tpr))
            #cmp = np.array(list(map(int, (pred != train_y[test_index]))))
            #e = np.dot(np.ones(pred.shape[0]), cmp)
            #err += (e / pred.shape[0])
            #print("err: ", e / pred.shape[0])
        auc = auc / 5
        ###
        draw_y.append(auc)
        ###
        #err = err / 5
        if auc > max_auc:
            max_auc = auc
            learner_num = i
        print('{} learners,  AUC: {}'.format(i, auc))
    
    ###
    plt.plot(draw_x, draw_y)
    plt.show()
    ###

    print('cross validation: AUC: {}, learner num: {}'.format(max_auc, learner_num))
    return learner_num

        

if __name__ == "__main__":
    # 读取训练集和测试集
    train_raw, test_raw = load_data()

    # 预处理数据集
    train_x, train_y, test_x, test_y = preprocessing(train_raw, test_raw)

    # 交叉验证
    #T = 50
    #learner_num = cross_validation(train_x, train_y, test_x, test_y, T)
    learner_num = 29

    # 训练、测试
    pred = adaboost(train_x, train_y, test_x, test_y, learner_num)
    fpr, tpr, thresholds = metrics.roc_curve(test_y.astype('int'), pred)
    auc = metrics.auc(fpr, tpr)
    cmp = np.array(list(map(int, (pred != test_y))))
    err = np.dot(np.ones(pred.shape[0]), cmp)
    print("number of learners:", learner_num)
    print('AUC: {}, accuracy: {}'.format(auc, 1 - err / pred.shape[0]))