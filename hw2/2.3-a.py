import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import math
from numpy.core.fromnumeric import argmax, argmin

class LinearKernelSVM:

    def __init__(self,):
        # parameter initialization
        self.TrainAccList = []

    
    def readfile():
        # manually remove the data with "?" and manually remove the attribute "Sample code number"
        cancer = pd.read_csv("cancer/cancer.csv", header = None)
        cancer_np = cancer.to_numpy()
        size = len(cancer_np)
        trainingSize = size * 0.7
        validationSize = size * 0.1
        #testingSize = size * 0.2
        trainingData = cancer_np[0:int(trainingSize)]
        validationData = cancer_np[int(trainingSize):int(trainingSize) + int(validationSize)]
        testingData = cancer_np[int(trainingSize) + int(validationSize):]
        
        return (trainingData, validationData, testingData)

    def Entropy(data):
        label = data[:,-1]
        c4 = 0 # numbers of class =4
        c2 = 0 # numbers of class =2
        for i in range(len(label)):
            if label[i]==4:
                c4 += 1
            else: 
                c2 += 1
        p4 = c4/len(label) # prob of class = 4
        p2 = c2/len(label) # prob of class = 2
        entropy = (-1 * p4 * math.log(p4,2)) + (-1* p2 * math.log(p2,2))
        return entropy

    training, validation, testing = readfile()
    print(Entropy(training))


# information gain for training datat = pd.read_csv("cancer/cancer.csv", header = None)

# t = pd.read_csv("cancer/cancer.csv", header = None)
# Class = t.keys()[-1]   #To make the code generic, changing target variable class name
# target_variables = t[Class].unique()  #This gives all '4' and '2'
# variables = t[1].unique() # 1-10
# ET = 0
# GIList = [] #gain information list
# list = []
# varList = []


# for i in range(1,10):
#     S = entorpy(y_np)
#     for variable in variables:
#         total = 0
#         t4 = 0
#         t2 = 0
#         for target_variable in target_variables:
#             num = len(t[i][t[i]==variable][t[Class] ==target_variable])
#             print(i,variable,target_variable,num)
#             if target_variable == 4:
#                 t4 = num
#             elif target_variable == 2:
#                 t2 = num
#             total += num
#         if t4 == 0 :
#             ET = 0
#         elif t2 == 0 :
#             ET = 0
#         else:
#             p4 = t4/total
#             p2 = t2/total
#             ET = (-1 * p4 * math.log(p4,2)) + (-1* p2 * math.log(p2,2))
#         pp = (t4+t2)/len(t)
#         S -= pp*ET
#     GIList.append(S)
#     print(S)
#     print(GIList)
#     print(argmax(GIList))


# for variable in variables:
#         total = 0
#         t4 = 0
#         t2 = 0
#         for target_variable in target_variables:
#             num = len(t[2][t[2]==variable][t[Class] ==target_variable])
#             print(variable,target_variable,num)
#             if target_variable == 4:
#                 t4 = num
#             elif target_variable == 2:
#                 t2 = num
#             total += num
#         if t4 == 0 :
#             ET = 0
#         elif t2 == 0 :
#             ET = 0
#         else:
#             p4 = t4/total
#             p2 = t2/total
#             ET = (-1 * p4 * math.log(p4,2)) + (-1* p2 * math.log(p2,2))
#         pp = (t4+t2)/len(t)
#         list.append(ET)
#         varList.append(variable)
# print(list)
# print(argmax(list))
# print(argmin(list))
# print(varList[argmax(list)])
# print(varList[argmin(list)])
