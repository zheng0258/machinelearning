import numpy as np
import matplotlib.pyplot as plt
import time
import mnist_reader
from sklearn import svm
import math

class LinearKernelSVM:

    def __init__(self,):
        # parameter initialization
        self.TrainAccList = []
        self.VaildAccList = []
        self.TestAccList = []

    
    def readfile():
        X_train, Y_train = mnist_reader.load_mnist('data', kind='train')
        X_test, Y_test = mnist_reader.load_mnist('data', kind='t10k')
        #size = len(X_train)
        #trainingSize = size * 0.8
        # trainingData = X_train[0:int(trainingSize)]
        # testingData = X_train[int(trainingSize):]
        # trainingLabel = Y_train[0:int(trainingSize)]
        # testingLabel = Y_train[int(trainingSize):]]
        trainingData = X_train[0:4800]
        validationData = X_train[4800:6000]
        trainingLabel = Y_train[0:4800]
        validationLabel = Y_train[4800:6000]
        return (trainingData, trainingLabel, validationData, validationLabel, X_test, Y_test)

    def SVMAccuracy(x_train, y_train, x_vaild, y_vaild, x_test, y_test):
        train_AccList = [] 
        vaild_AccList = []
        test_AccList = []
        C_List = []
        c = math.pow(10, -4)
        for i in range(9):
            svc = svm.SVC(kernel ='linear', C = c).fit(x_train, y_train)
            print(c)
            C_List.append(c)
            train_AccList.append(svc.score(x_train,y_train))
            vaild_AccList.append(svc.score(x_vaild,y_vaild))
            test_AccList.append(svc.score(x_test,y_test))
            c *= 10
            i += 1
        return train_AccList, vaild_AccList, test_AccList, C_List

    def plot(train_AccList, vaild_AccList, test_AccList, C_List):
        paraTick = range(len(C_List))
        plt.plot(paraTick,train_AccList, '-o',label = 'Training Accuracy', alpha=0.3)
        plt.plot(paraTick,vaild_AccList, '-o',label = 'Validation Accuracy', alpha=0.3)
        plt.plot(paraTick,test_AccList, '-o',label = 'Testing Accuracy', alpha=0.3)
        plt.xticks(paraTick,C_List)
        plt.xlabel('C parameter')
        plt.ylabel('Accuracy')
        plt.title('Linear Kernel SVM')
        plt.legend()
        plt.draw()
        plt.savefig('2.1-a.jpg')

    x_train, y_train, x_vaild, y_vaild, x_test, y_test = readfile()
    train_AccList, vaild_AccList, test_AccList, C_List = SVMAccuracy(x_train, y_train, x_vaild, y_vaild, x_test, y_test)
    print(train_AccList)
    print(vaild_AccList)
    print(test_AccList)
    print(C_List)
    plot(train_AccList, vaild_AccList, test_AccList, C_List)





