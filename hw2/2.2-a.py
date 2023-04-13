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
        X_train, Y_train = mnist_reader.load_mnist('data', kind='train')
        X_test, Y_test = mnist_reader.load_mnist('data', kind='t10k')
        size = len(X_train)
        trainingSize = size * 0.8
        trainingData = X_train[0:int(trainingSize)]
        validationData = X_train[int(trainingSize):]
        trainingLabel = Y_train[0:int(trainingSize)]
        validationLabel = Y_train[int(trainingSize):]
        return (trainingData, trainingLabel, validationData, validationLabel, X_test, Y_test)

    def KernelizedPerceptronD2(iterations, x_train, y_train, x_vaild, y_vaild, x_test, y_test):
        y_pred = 0
        w = np.zeros([10,1])
        TrainingErrlist = []
        VaildErr = 0
        TestErr = 0
        result = []
        for i in range(iterations):
            mistake = 0
            for j in range(len(x_train)):
                kerneled_data = (1 + np.linalg.norm(x_train[j]))**2
                form_train = formXY(kerneled_data,y_train[j])
                classNo = y_train[j]
                argmax = np.dot(w, form_train[classNo]) #predict using the current weight
                y_pred = np.argmax(argmax)  #predict
                if y_pred != y_train[j]:  #if mistake    
                    #update wieght vecotrs
                    represent2XY = formXY(kerneled_data,y_train[j])
                    represent2XY_pred = formXY(kerneled_data,y_pred)
                    w[y_train[j]] += represent2XY[y_train[j]]
                    w[y_pred] -= represent2XY_pred[y_pred]
                    mistake +=1
            TrainingErrlist.append(mistake) 

        for k in range(len(x_vaild)):
            kerneled_data = (1 + np.linalg.norm(x_vaild[k]))**2
            form_vaild = formXY(kerneled_data,y_vaild[k])
            classNo = y_vaild[k]
            argmax = np.dot(w, form_vaild[classNo]) #predict using the current weight
            y_pred = np.argmax(argmax)  #predict
            if y_pred != y_vaild[k]:  #if mistake    
                VaildErr += 1
        for m in range(len(x_test)):
            kerneled_data = ( 1 + np.linalg.norm(x_test[m]))**2
            form_test = formXY(kerneled_data,y_test[m])
            classNo = y_test[m]
            argmax = np.dot(w, form_test[classNo]) #predict using the current weight
            y_pred = np.argmax(argmax)  #predict
            if y_pred != y_test[m]:  #if mistake    
                TestErr += 1
        result.append((len(x_train)-mistake)/len(x_train))  #result[0]=training accuracy
        result.append((len(x_vaild)-VaildErr)/len(x_vaild)) #result[1]=vaildation accuracy
        result.append((len(x_test)-TestErr)/len(x_test))    #result[3]=testing accuracy

        return TrainingErrlist, result
    
    def KernelizedPerceptronD3(iterations, x_train, y_train, x_vaild, y_vaild, x_test, y_test):
        y_pred = 0
        w = np.zeros([10,1])
        TrainingErrlist = []
        VaildErr = 0
        TestErr = 0
        result = []
        for i in range(iterations):
            mistake = 0
            for j in range(len(x_train)):
                kerneled_data = ( 1 + np.linalg.norm(x_train[j]))**3
                form_train = formXY(kerneled_data,y_train[j])
                classNo = y_train[j]
                argmax = np.dot(w, form_train[classNo]) #predict using the current weight
                y_pred = np.argmax(argmax)  #predict
                if y_pred != y_train[j]:  #if mistake    
                    #update wieght vecotrs
                    represent2XY = formXY(kerneled_data,y_train[j])
                    represent2XY_pred = formXY(kerneled_data,y_pred)
                    w[y_train[j]] += represent2XY[y_train[j]]
                    w[y_pred] -= represent2XY_pred[y_pred]
                    mistake +=1
            TrainingErrlist.append(mistake) 

        for k in range(len(x_vaild)):
            kerneled_data = ( 1 + np.linalg.norm(x_vaild[k]))**3
            form_vaild = formXY(kerneled_data,y_vaild[k])
            classNo = y_vaild[k]
            argmax = np.dot(w, form_vaild[classNo]) #predict using the current weight
            y_pred = np.argmax(argmax)  #predict
            if y_pred != y_vaild[k]:  #if mistake    
                VaildErr += 1
        for m in range(len(x_test)):
            kerneled_data = ( 1 + np.linalg.norm(x_test[m]))**3
            form_test = formXY(kerneled_data,y_test[m])
            classNo = y_test[m]
            argmax = np.dot(w, form_test[classNo]) #predict using the current weight
            y_pred = np.argmax(argmax)  #predict
            if y_pred != y_test[m]:  #if mistake    
                TestErr += 1
        result.append((len(x_train)-mistake)/len(x_train))  #result[0]=training accuracy
        result.append((len(x_vaild)-VaildErr)/len(x_vaild)) #result[1]=vaildation accuracy
        result.append((len(x_test)-TestErr)/len(x_test))    #result[3]=testing accuracy

        return TrainingErrlist, result

    def KernelizedPerceptronD4(iterations, x_train, y_train, x_vaild, y_vaild, x_test, y_test):
        y_pred = 0
        w = np.zeros([10,1])
        TrainingErrlist = []
        VaildErr = 0
        TestErr = 0
        result = []
        for i in range(iterations):
            mistake = 0
            for j in range(len(x_train)):
                kerneled_data = ( 1 + np.linalg.norm(x_train[j]))**4
                form_train = formXY(kerneled_data,y_train[j])
                classNo = y_train[j]
                argmax = np.dot(w, form_train[classNo]) #predict using the current weight
                y_pred = np.argmax(argmax)  #predict
                if y_pred != y_train[j]:  #if mistake    
                    #update wieght vecotrs
                    represent2XY = formXY(kerneled_data,y_train[j])
                    represent2XY_pred = formXY(kerneled_data,y_pred)
                    w[y_train[j]] += represent2XY[y_train[j]]
                    w[y_pred] -= represent2XY_pred[y_pred]
                    mistake +=1
            TrainingErrlist.append(mistake) 
        for k in range(len(x_vaild)):
            kerneled_data = ( 1 + np.linalg.norm(x_vaild[k]))**4
            form_vaild = formXY(kerneled_data,y_vaild[k])
            classNo = y_vaild[k]
            argmax = np.dot(w, form_vaild[classNo]) #predict using the current weight
            y_pred = np.argmax(argmax)  #predict
            if y_pred != y_vaild[k]:  #if mistake    
                VaildErr += 1
        for m in range(len(x_test)):
            kerneled_data = ( 1 + np.linalg.norm(x_test[m]))**4
            form_test = formXY(kerneled_data,y_test[m])
            classNo = y_test[m]
            argmax = np.dot(w, form_test[classNo]) #predict using the current weight
            y_pred = np.argmax(argmax)  #predict
            if y_pred != y_test[m]:  #if mistake    
                TestErr += 1
        result.append((len(x_train)-mistake)/len(x_train))  #result[0]=training accuracy
        result.append((len(x_vaild)-VaildErr)/len(x_vaild)) #result[1]=vaildation accuracy
        result.append((len(x_test)-TestErr)/len(x_test))    #result[3]=testing accuracy

        return TrainingErrlist, result

    def plot(iterations, D2errList, D3errList, D4errList):
        plt.plot(np.arange(iterations),D2errList, '-o', label = 'Degree = 2')
        plt.plot(np.arange(iterations),D3errList, '-o', label = 'Degree = 3')
        plt.plot(np.arange(iterations),D4errList, '-o', label = 'Degree = 4')
        plt.xlabel('Number of Training Iterations')
        plt.ylabel('Number of Mistakes')
        plt.title('Online Learning Curve')
        plt.legend()
        plt.draw()
        plt.savefig('2.2-aw.jpg')
    
    def printAcc(title, result):
        print(title + " Training Accuracy: " + str(result[0]))
        print(title + " Vaildation Accuracy: " + str(result[1]))
        print(title + " Testing Accuracy:" + str(result[2]))

    iterations = 5
    x_train, y_train, x_vaild, y_vaild, x_test, y_test = readfile()
    D2errList, D2result = KernelizedPerceptronD2(iterations, x_train, y_train, x_vaild, y_vaild, x_test, y_test)
    D3errList, D3result = KernelizedPerceptronD3(iterations, x_train, y_train, x_vaild, y_vaild, x_test, y_test)
    D4errList, D4result = KernelizedPerceptronD4(iterations, x_train, y_train, x_vaild, y_vaild, x_test, y_test)
    plot(iterations, D2errList, D3errList, D4errList)
    printAcc("Degree = 2", D2result)
    printAcc("Degree = 3", D3result)
    printAcc("Degree = 4", D4result)




