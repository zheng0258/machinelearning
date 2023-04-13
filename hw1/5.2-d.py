import numpy as np
import matplotlib.pyplot as plt
import time
import mnist_reader

def formXY(x,y):
    classForm = [0] * 10
    for i in range(10): #class number is 10
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
        test_correctNum = 0
        test_correctList = []
        examplelist = []
        exampleNumber = 100

        for i in range(iterations):
            test_correctNum = 0
            # training
            for j in range(exampleNumber):
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

            examplelist.append(exampleNumber)
            exampleNumber += 100 
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

        return test_correctList, examplelist

    def PassiveAgreessive(iterations, x_train, y_train, x_test, y_test):
        tau = 0
        y_pred = 0
        w = np.zeros([10,784])
        temp = np.zeros([10,784])
        test_correctNum = 0
        test_correctList = []
        examplelist = []
        exampleNumber = 100

        for i in range(iterations): # for each training iteration    
            test_correctNum = 0  
            # training
            for j in range(exampleNumber):
                form_train = formXY(x_train[j],y_train[j])
                classNo = y_train[j]
                argmax = np.dot(w, form_train[classNo])
                y_pred = np.argmax(argmax)
                if y_pred != y_train[j]:
                    #compute tau
                    form_pred = formXY(x_train[j],y_pred)
                    wFxy = np.dot(w[classNo], form_train[classNo])
                    wFxy_pred = np.dot(w[y_pred], form_pred[y_pred])
                    numerator =  1 -(wFxy - wFxy_pred)

                    temp[classNo] = form_train[classNo]
                    temp[y_pred] -= form_pred[y_pred]

                    demoninator = np.linalg.norm(temp)**2
                    tau = numerator / demoninator
                    #update wieght vecotrs
                    represent2XY = formXY(x_train[j],y_train[j])
                    represent2XY_pred = formXY(x_train[j],y_pred)
                    w[y_train[j]] += tau * represent2XY[y_train[j]]
                    w[y_pred] -= tau * represent2XY_pred[y_pred]

            examplelist.append(exampleNumber)
            exampleNumber += 100
            # testing
            for j in range(len(x_test)):
                form_test = formXY(x_test[j],y_test[j])
                classNo = y_test[j]
                argmax = np.dot(w, form_test[classNo])
                y_pred = np.argmax(argmax)
                if y_pred != y_test[j]:
                    #compute tau
                    form_pred = formXY(x_test[j],y_pred)
                    wFxy = np.dot(w[classNo], form_test[classNo])
                    wFxy_pred = np.dot(w[y_pred], form_pred[y_pred])
                    numerator =  1 -(wFxy - wFxy_pred)

                    temp[classNo] = form_test[classNo]
                    temp[y_pred] -= form_pred[y_pred]

                    demoninator = np.linalg.norm(temp)**2
                    tau = numerator / demoninator
                    #update wieght vecotrs
                    represent2XY = formXY(x_test[j],y_test[j])
                    represent2XY_pred = formXY(x_test[j],y_pred)
                    w[y_test[j]] += tau * represent2XY[y_test[j]]
                    w[y_pred] -= tau * represent2XY_pred[y_pred]
                else:
                    test_correctNum +=1
            test_correctList.append(test_correctNum / len(x_test))          
        return test_correctList, examplelist

    def plot(iterations, PerceptronTestAccList, PerceptronExampleList, PassiveAgreessiveTestAccList, PassiveAgressiveExampleList):
        plt.plot(PerceptronExampleList,PerceptronTestAccList, '-o',label = 'Perceptron Test', alpha=0.3)
        plt.plot(PassiveAgressiveExampleList,PassiveAgreessiveTestAccList, '-o',label = 'PassiveAgressive Test', alpha=0.3)
        plt.xlabel('Number of Examples')
        plt.ylabel('Testing Accuracy')
        plt.title('General Learning Curve')
        plt.legend()
        plt.draw()
        plt.savefig('5.2-d.jpg')


    iterations = 20
    x_train, y_train, x_test, y_test = readfile()


    PerceptronTestAccList, PerceptronExamplelist = Perceptron(iterations, x_train, y_train, x_test, y_test)
    PassiveAgreessiveTestAccList, PassiveAgreessiveExamplelist = PassiveAgreessive(iterations, x_train, y_train, x_test, y_test)
    plot(iterations, PerceptronTestAccList, PerceptronExamplelist, PassiveAgreessiveTestAccList, PassiveAgreessiveExamplelist)




