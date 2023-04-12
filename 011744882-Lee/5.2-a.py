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
        return (X_train, y_train)

    def Perceptron(iterations, x, y):
        y_pred = 0
        w = np.zeros([10,784])
        errlist = []
        for i in range(50):
            mistake = 0
            for j in range(len(x)):

                form_train = formXY(x[j],y[j])
                classNo = y[j]
                argmax = np.dot(w, form_train[classNo]) #predict using the current weight
                y_pred = np.argmax(argmax)  #predict
                if y_pred != y[j]:  #if mistake    
                    #update wieght vecotrs
                    represent2XY = formXY(x[j],y[j])
                    represent2XY_pred = formXY(x[j],y_pred)
                    w[y[j]] += represent2XY[y[j]]
                    w[y_pred] -= represent2XY_pred[y_pred]
                    mistake +=1
            errlist.append(mistake) 
        return errlist

    def PassiveAgreessive(iterations, x, y):
        tau = 0
        y_pred = 0
        w = np.zeros([10,784])
        errlist = []
        temp = np.zeros([10,784])

        for i in range(iterations): # for each training iteration
            mistake = 0
            for j in range(len(x)):
                form_train = formXY(x[j],y[j])
                classNo = y[j]
                argmax = np.dot(w, form_train[classNo])
                y_pred = np.argmax(argmax)
                if y_pred != y[j]:
                    #compute tau
                    form_pred = formXY(x[j],y_pred)
                    wFxy = np.dot(w[classNo], form_train[classNo])
                    wFxy_pred = np.dot(w[y_pred], form_pred[y_pred])
                    numerator =  1 -(wFxy - wFxy_pred)

                    temp[classNo] = form_train[classNo]
                    temp[y_pred] -= form_pred[y_pred]

                    demoninator = np.linalg.norm(temp)**2
                    tau = numerator / demoninator
                    #update wieght vecotrs
                    represent2XY = formXY(x[j],y[j])
                    represent2XY_pred = formXY(x[j],y_pred)
                    w[y[j]] += tau * represent2XY[y[j]]
                    w[y_pred] -= tau * represent2XY_pred[y_pred]
                    mistake +=1
            errlist.append(mistake)         
        return errlist

    def plot(iterations, PerceptronErrlist, PassiveAgreessiveErrlist):
        plt.plot(np.arange(iterations),PerceptronErrlist, '-o', label = 'Perceptron')
        plt.plot(np.arange(iterations),PassiveAgreessiveErrlist, '-o', label = 'PassiveAgressive')
        plt.xlabel('Number of Training Iterations')
        plt.ylabel('Number of Mistakes')
        plt.title('Online Learning Curve')
        plt.legend()
        plt.draw()
        plt.savefig('5.2-a.jpg')

    iterations = 50
    x,y = readfile()

    PerceptronErrlist = Perceptron(iterations, x, y)
    PassiveAgreessiveErrlist = PassiveAgreessive(iterations, x, y)
    plot(iterations, PerceptronErrlist, PassiveAgreessiveErrlist)





