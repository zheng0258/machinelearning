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
        c = math.pow(10, -3)
        svcP2 = svm.SVC(kernel ='poly', degree= 2, C = c).fit(x_train, y_train)
        print("polynomial kernel of degree2")
        print("training accuracy:")
        print(svcP2.score(x_train,y_train))
        print("vaildation accuracy:")
        print(svcP2.score(x_vaild,y_vaild))
        print("testing accuracy: ")
        print(svcP2.score(x_test,y_test))
        print("the number of support vectors: ")
        print(svcP2._n_support)

        svcP3 = svm.SVC(kernel ='poly', degree= 3, C = c).fit(x_train, y_train)
        print("polynomial kernel of degree 3")
        print("training accuracy:")
        print(svcP3.score(x_train,y_train))
        print("vaildation accuracy:")
        print(svcP3.score(x_vaild,y_vaild))
        print("testing accuracy: ")
        print(svcP3.score(x_test,y_test))
        print("the number of support vectors: ")
        print(svcP3._n_support)

        svcP4 = svm.SVC(kernel ='poly', degree= 4, C = c).fit(x_train, y_train)
        print("polynomial kernel of degree 4")
        print("training accuracy:")
        print(svcP4.score(x_train,y_train))
        print("vaildation accuracy:")
        print(svcP4.score(x_vaild,y_vaild))
        print("testing accuracy: ")
        print(svcP4.score(x_test,y_test))
        print("the number of support vectors: ")
        print(svcP4._n_support)


    x_train, y_train, x_vaild, y_vaild, x_test, y_test = readfile()
    SVMAccuracy(x_train, y_train, x_vaild, y_vaild, x_test, y_test)




