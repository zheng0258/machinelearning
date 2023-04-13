import numpy as np
import matplotlib.pyplot as plt
import time
import mnist_reader

class BincaryClassification:

    def __init__(self,):
        # parameter initialization
        self.iterations = 0
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
        test_correctNum = 0
        test_correctList = []
        w = [0]*len(x_train[0])
        y_pred = 0
        exampleList = []
        exampleNumber = 100

        for i in range(iterations): # for each training iteration
            test_correctNum = 0
            for j in range(exampleNumber):
                y_pred = np.sign(np.dot(w,x_train[j]))
                if y_pred != y_train[j]:
                    w = np.add(w,tau * y_train[j] * x_train[j])
            exampleList.append(exampleNumber)
            exampleNumber += 100
            for j in range(len(x_test)):
                y_pred = np.sign(np.dot(w,x_test[j]))
                if y_pred != y_test[j]:
                    w = np.add(w,tau * y_test[j] * x_test[j])
                else:
                    test_correctNum +=1
            test_correctList.append(test_correctNum/len(x_test))
        return test_correctList, exampleList

    def PassiveAgreessive(iterations, x_train, y_train, x_test, y_test):
        tau = 1
        test_correctNum = 0
        test_accList = []
        w = [0]*len(x_train[0])
        y_pred = 0
        numerator =  0
        demoninator = 0
        examplelist = []
        exampleNumber = 100

        for i in range(iterations): # for each training iteration
            test_correctNum = 0
            for j in range(exampleNumber):
                y_pred = np.sign(np.dot(w,x_train[j]))
                if y_pred != y_train[j]:
                    numerator =  1 -(y_train[j] * (np.dot(w, x_train[j])))
                    demoninator = np.linalg.norm(x_train[j])**2
                    tau = numerator / demoninator
                    tau = max(0, tau)
                    w = np.add(w, tau * y_train[j] * x_train[j])
            examplelist.append(exampleNumber)
            exampleNumber += 100
            for j in range(len(x_test)):
                y_pred = np.sign(np.dot(w,x_test[j]))
                if y_pred != y_test[j]:
                    numerator =  1 -(y_test[j] * (np.dot(w, x_test[j])))
                    demoninator = np.linalg.norm(x_test[j])**2
                    tau = numerator / demoninator
                    tau = max(0, tau)
                    w = np.add(w, tau * y_test[j] * x_test[j])
                else:
                    test_correctNum += 1  
            test_accList.append(test_correctNum/len(x_test))    
        return test_accList, examplelist

    def plot(iterations, PerceptronTestAccList, PerceptronExampleList, PassiveAgreessiveTestAccList, PassiveAgressiveExampleList):
        plt.plot(PerceptronExampleList,PerceptronTestAccList, '-o',label = 'Perceptron Test', alpha=0.3)
        plt.plot(PassiveAgressiveExampleList,PassiveAgreessiveTestAccList, '-o',label = 'PassiveAgressive Test', alpha=0.3)
        plt.xlabel('Number of Examples')
        plt.ylabel('Testing Accuracy')
        plt.title('General Learning Curve')
        plt.legend()
        plt.draw()
        plt.savefig('5.1-d.jpg')

    iterations = 20
    x_train, y_train, x_test, y_test = readfile()
    y_train = even_odd_label(y_train)
    y_test = even_odd_label(y_test)


    PerceptronTestAccList, PerceptronExamplelist = Perceptron(iterations, x_train, y_train, x_test, y_test)
    PassiveAgreessiveTestAccList, PassiveAgreessiveExamplelist = PassiveAgreessive(iterations, x_train, y_train, x_test, y_test)
    plot(iterations, PerceptronTestAccList, PerceptronExamplelist, PassiveAgreessiveTestAccList, PassiveAgreessiveExamplelist)





