import numpy as np
import matplotlib.pyplot as plt
import time
import mnist_reader

class BincaryClassification:

    def __init__(self,):
        # parameter initialization
        self.iterations = 0
        self.errlist = []
        self.PerceptronErrlist = [] 
        self.PassiveAgreessiveErrlist = []
        self.x,self.y, self.y_label = []

    
    def readfile():
        X_train, y_train = mnist_reader.load_mnist('data', kind='train')
        return (X_train, y_train)

    def even_odd_label(train_label):
        y=[0.0] * len(train_label)

        for i in range(len(train_label)):
            if train_label[i]%2 == 0:
                y[i] = 1.0
            else:
                y[i] = -1
        return y

    def Perceptron(iterations, x, y):
        tau = 1
        errlist = []
        w = [0]*len(x[0])
        y_pred = 0

        for i in range(iterations): # for each training iteration
            mistake = 0
            for j in range(len(x)):
                y_pred = np.sign(np.dot(w,x[j]))
                if y_pred != y[j]:
                    w = np.add(w,tau * y[j] * x[j])
                    mistake += 1
            errlist.append(mistake)  

        return errlist

    def PassiveAgreessive(iterations, x, y):
        tau = 0
        errlist = [] # error list
        w = [0]*len(x[0]) # weight
        y_pred = 0 # predicted y label
        numerator =  0
        demoninator = 0

        for i in range(iterations): # for each training iteration
            mistake = 0
            for j in range(len(x)):
                y_pred = np.sign(np.dot(w,x[j]))
                if y_pred != y[j]:
                    numerator =  1 -(y[j] * (np.dot(w, x[j])))
                    demoninator = np.linalg.norm(x[j])**2
                    tau = numerator / demoninator
                    tau = max(0, tau)
                    w = np.add(w, tau * y[j] * x[j])
                    mistake += 1
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
        plt.savefig('5.1-a.jpg')

    iterations = 50
    x,y = readfile()
    label_y = even_odd_label(y)

    PerceptronErrlist = Perceptron(iterations, x, label_y)
    PassiveAgreessiveErrlist = PassiveAgreessive(iterations, x, label_y)
    plot(iterations, PerceptronErrlist, PassiveAgreessiveErrlist)





