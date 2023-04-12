import numpy as np
import matplotlib.pyplot as plt
import time
import mnist_reader

class BincaryClassification:

    def __init__(self,):
        # parameter initialization
        self.iterations = 0
        self.acclist = []
        self.PerceptronAcclist = [] 
        self.PassiveAgreessiveAcclist = []
        self.x,self.y, self.y_label = []

    
    def readfile():
        X_train, y_train = mnist_reader.load_mnist('data', kind='train')
        X_test, y_test = mnist_reader.load_mnist('data', kind='t10k')
        return (X_train, y_train, X_test, y_test)

    def even_odd_label(train_label):
        y=[0.0] * len(train_label)

        for i in range(len(train_label)):
            if train_label[i]%2 == 0:
                y[i] = 1.0
            else:
                y[i] = -1
        return y

    def Perceptron(iterations, x_train, y_train, x_test, y_test):
        tau = 1
        train_correctNum = 0
        train_correctList = []
        test_correctNum = 0
        test_correctList = []
        w = [0]*len(x_train[0])
        y_pred = 0

        for i in range(iterations): # for each training iteration
            train_correctNum = 0
            test_correctNum = 0
            for j in range(len(x_train)):
                y_pred = np.sign(np.dot(w,x_train[j]))
                if y_pred != y_train[j]:
                    w = np.add(w,tau * y_train[j] * x_train[j])
                else:
                    train_correctNum +=1
            train_correctList.append(train_correctNum/len(x_train))

            for j in range(len(x_test)):
                y_pred = np.sign(np.dot(w,x_test[j]))
                if y_pred != y_test[j]:
                    w = np.add(w,tau * y_test[j] * x_test[j])
                else:
                    test_correctNum +=1
            test_correctList.append(test_correctNum/len(x_test))
        return train_correctList, test_correctList

    def AveragedPerceptron(iterations, x_train, y_train, x_test, y_test):
        tau = 1
        train_correctNum = 0
        train_correctList = []
        test_correctNum = 0
        test_correctList = []
        count = 0
        w = [0]*len(x_train[0])
        wSum = [0]*len(x_train[0])
        wAvg = [0]*len(x_train[0])
        y_pred = 0

        for i in range(iterations): # for each training iteration
            train_correctNum = 0
            for j in range(len(x_train)):
                y_pred = np.sign(np.dot(w,x_train[j]))
                if y_pred != y_train[j]:
                    w = w + tau * np.dot(x_train[j], y_train[j])
                    temp = np.dot(tau, y_train[j])
                    wSum = wSum + np.dot(temp, x_train[j])
                    count += 1
                else:
                    train_correctNum +=1
            train_correctList.append(train_correctNum/len(x_train))
        
        wAvg = wSum / count

        for i in range(iterations):
            test_correctNum = 0
            for j in range(len(x_test)):
                y_pred = np.sign(np.dot(wAvg,x_test[j]))
                if y_pred != y_test[j]:
                    wAvg = np.add(wAvg,tau * y_test[j] * x_test[j])
                else:
                    test_correctNum +=1
            test_correctList.append(test_correctNum/len(x_test))
        return train_correctList, test_correctList

    def plot(iterations, PerceptronTrainAccList, PerceptronTestAccList, AveragedPerceptronTrainAccList, AveragedPerceptronTestAccList):
        plt.plot(np.arange(iterations),PerceptronTrainAccList, '-o',label = 'Perceptron Train', alpha=0.3)
        plt.plot(np.arange(iterations),PerceptronTestAccList, '-o',label = 'Perceptron Test', alpha=0.3)
        plt.plot(np.arange(iterations),AveragedPerceptronTrainAccList, '-o',label = 'AveragedPerceptron Train', alpha=0.3)
        plt.plot(np.arange(iterations),AveragedPerceptronTestAccList, '-o',label = 'AveragedPerceptron Test', alpha=0.3)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.draw()
        plt.savefig('5.1-c.jpg')

    iterations = 20
    x_train, y_train, x_test, y_test = readfile()
    y_train = even_odd_label(y_train)
    y_test = even_odd_label(y_test)

    PerceptronTrainAccList, PerceptronTestAccList = Perceptron(iterations, x_train, y_train, x_test, y_test)
    AveragedPerceptronTrainAccList, AveragedPerceptronTestAccList = AveragedPerceptron(iterations, x_train, y_train, x_test, y_test)
    plot(iterations, PerceptronTrainAccList, PerceptronTestAccList, AveragedPerceptronTrainAccList, AveragedPerceptronTestAccList)




