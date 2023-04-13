import numpy as np
import pandas as pd

class NavieBayes:

    def __init__(self,):
        self.stop_words = 0

    
    def readfile():
        stop_words = []
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        stoplist = open('fortune-cookie-data/stoplist.txt')
        traindata = open('fortune-cookie-data/traindata.txt')
        trainlabel = open('fortune-cookie-data/trainlabels.txt')
        testdata = open('fortune-cookie-data/testdata.txt')
        testlabel = open('fortune-cookie-data/testlabels.txt')
        
        for i in stoplist:
            i = i.replace('\n', '')
            stop_words.append(i)
            
        for i in traindata:
            i = i.replace('\n', '')
            x_train.append(i)
            
        for i in trainlabel:
            i = i.replace('\n', '')
            y_train.append(int(i))
        
        for i in testdata:
            i = i.replace('\n', '')
            x_test.append(i)
        
        for i in testlabel:
            i = i.replace('\n', '')
            y_test.append(int(i))

        return (stop_words, x_train, x_test, y_train, y_test)

    def preprocess(stop_words, x_train):
        
        vocabulary = []
        
        for i in x_train:
            i = i.replace('\n', '')
            i = i.split(' ')
            for word in i:
                if word not in stop_words and word not in vocabulary and len(word) > 0:
                    vocabulary.append(word)
        vocabulary.sort() # alphabetical order
        print(vocabulary)
        return vocabulary

    def features(x_train, y_train, x_test, y_test,vocabulary):
        M = len(vocabulary) # the size of vocabulary
        train_vector = np.zeros((len(x_train), M)) #feature vector of size M
        test_vector = np.zeros((len(x_test), M)) #feature vector of size M
        train_count = 0
        test_count = 0

        for i in x_train:
            i = i.replace('\n', '')
            i = i.split(' ')
            for word in i:
                if word in vocabulary:
                    index = vocabulary.index(word)
                    train_vector[train_count][index] = 1      
            train_count += 1
        for i in x_test:
            i = i.replace('\n', '')
            i = i.split(' ')
            for word in i:
                if word in vocabulary:
                    index = vocabulary.index(word)
                    test_vector[test_count][index] = 1        
            test_count += 1

        train_vector = pd.DataFrame(train_vector, columns = vocabulary)
        test_vector = pd.DataFrame(test_vector, columns = vocabulary)

        train_label = pd.DataFrame(y_train, columns = ['label']) #add column label
        test_label =  pd.DataFrame(y_test, columns = ['label']) #add column label

        return (train_vector, train_label, test_vector, test_label)


    # def NaiveBayesClassifier(train_vector, train_label, test_vector, test_label,vocabulary):
        
    #     return 1


    stop_words, x_train, x_test, y_train, y_test = readfile()
    vocabulary = preprocess(stop_words, x_train)
    train_vector, train_label, test_vector, test_label = features(x_train, y_train, x_test, y_test, vocabulary)
    #NaiveBayesClassifier(train_vector, train_label, test_vector, test_label, vocabulary)







