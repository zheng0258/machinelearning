import numpy as np
import matplotlib.pyplot as plt
import time
import mnist_reader
from sklearn import svm
from sklearn.metrics import accuracy_score
import math
import pandas as pd


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
        combinedData = X_train[0:6000]
        combinedLabel = Y_train[0:6000]
        return (combinedData, combinedLabel, X_test, Y_test)

    def TestAccuracyAndConfusionMatrix(x_combined, y_combined, x_test, y_test):
        c = math.pow(10, -3)
        svc = svm.SVC(kernel ='linear', C = c).fit(x_combined, y_combined)
        print(svc.score(x_test,y_test))
        y_pred = svc.predict(x_test)
        confusion_matrix = pd.crosstab(y_test,y_pred, rownames=['Actual'], colnames=['Predicted'])
        print(confusion_matrix)


    x_combined, y_combined, x_test, y_test = readfile()
    TestAccuracyAndConfusionMatrix(x_combined, y_combined, x_test, y_test)






