import numpy as np
import matplotlib.pyplot as plt
import time
import mnist_reader

def formXY(x,y):
    classForm = [0] * 10
    for i in range(10): #class numver is 10
        if i == y:
            classForm[i]=x
    return classForm

class MultiClassClassification:

    def __init__(self,):
        # parameter initialization
        self.iterations = 0
        self.errlist = []
        self.PerceptronErrlist = [] 
        self.PassiveAgreessiveErrlist = []
        self.x, self.y, self.y_label = []

    
    def readfile():
        X_train, y_train = mnist_reader.load_mnist('data', kind='train')
        X_test, y_test = mnist_reader.load_mnist('data', kind='t10k')
        return (X_train, y_train, X_test, y_test)

    def Perceptron(iterations, x_train, y_train, x_test, y_test):
        y_pred = 0
        w = np.zeros([10,784])
        train_correctNum = 0
        train_correctList = []
        test_correctNum = 0
        test_correctList = []
        for i in range(iterations):
            train_correctNum = 0
            test_correctNum = 0
            # training
            for j in range(len(x_train)):
                form_train = formXY(x_train[j],y_train[j])
                classNo = y_train[j]
                argmax = np.dot(w, form_train[classNo]) #predict using the current weight
                y_pred = np.argmax(argmax)  #predict
                if y_pred != y_train[j]:  #if mistake    
                    #update wieght vecotrs
                    represent2XY = formXY(x_train[j],y_train[j])
                    represent2XY_pred = formXY(x_train[j],y_pred)
                    w[y_train[j]] += represent2XY[y_train[j]]
                    w[y_pred] -= represent2XY_pred[y_pred]
                else:
                    train_correctNum +=1
            train_correctList.append(train_correctNum/len(x_train))  
            # testing
            for j in range(len(x_test)):
                form_test = formXY(x_test[j],y_test[j])
                classNo = y_test[j]
                argmax = np.dot(w, form_test[classNo]) #predict using the current weight
                y_pred = np.argmax(argmax)  #predict
                if y_pred != y_test[j]:  #if mistake    
                    #update wieght vecotrs
                    represent2XY = formXY(x_test[j],y_test[j])
                    represent2XY_pred = formXY(x_test[j],y_pred)
                    w[y_test[j]] += represent2XY[y_test[j]]
                    w[y_pred] -= represent2XY_pred[y_pred]
                else:
                    test_correctNum +=1
            test_correctList.append(test_correctNum/len(x_test)) 

        return train_correctList, test_correctList

    def AveragedPerceptron(iterations, x_train, y_train, x_test, y_test):
        y_pred = 0
        w = np.zeros([10,784])
        wSum = np.zeros([10,784])
        wAvg = np.zeros([10,784])
        countList = [0]*10
        train_correctNum = 0
        train_correctList = []
        test_correctNum = 0
        test_correctList = []
        for i in range(iterations):
            train_correctNum = 0
            # training
            for j in range(len(x_train)):
                form_train = formXY(x_train[j],y_train[j])
                classNo = y_train[j]
                argmax = np.dot(w, form_train[classNo]) #predict using the current weight
                y_pred = np.argmax(argmax)  #predict
                if y_pred != y_train[j]:  #if mistake    
                    #update wieght vecotrs
                    represent2XY = formXY(x_train[j],y_train[j])
                    represent2XY_pred = formXY(x_train[j],y_pred)
                    w[y_train[j]] += represent2XY[y_train[j]]
                    w[y_pred] -= represent2XY_pred[y_pred]

                    wSum[y_train[j]] += represent2XY[y_train[j]]
                    countList[y_train[j]] +=1
                    wSum[y_pred] -= represent2XY_pred[y_pred]
                    countList[y_pred] +=1
                else:
                    train_correctNum +=1
            train_correctList.append(train_correctNum/len(x_train))  

        # averaged w
        for j in range(10):
            wAvg[j] = wSum[j] / countList[j]
            
        # testing
        for i in range(iterations):
            test_correctNum = 0
            for j in range(len(x_test)):
                form_test = formXY(x_test[j],y_test[j])
                classNo = y_test[j]
                argmax = np.dot(wAvg, form_test[classNo]) #predict using the current weight
                y_pred = np.argmax(argmax)  #predict
                if y_pred != y_test[j]:  #if mistake    
                    #update wieght vecotrs
                    represent2XY = formXY(x_test[j],y_test[j])
                    represent2XY_pred = formXY(x_test[j],y_pred)
                    wAvg[y_test[j]] += represent2XY[y_test[j]]
                    wAvg[y_pred] -= represent2XY_pred[y_pred]
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
        plt.savefig('5.2-c.jpg')

    iterations = 20
    x_train, y_train, x_test, y_test = readfile()

    PerceptronTrainAccList, PerceptronTestAccList = Perceptron(iterations, x_train, y_train, x_test, y_test)
    AveragedPerceptronTrainAccList, AveragedPerceptronTestAccList = AveragedPerceptron(iterations, x_train, y_train, x_test, y_test)
    plot(iterations, PerceptronTrainAccList, PerceptronTestAccList, AveragedPerceptronTrainAccList, AveragedPerceptronTestAccList)




