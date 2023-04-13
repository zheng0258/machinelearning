#%%

import DataPreprocess
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.inspection import permutation_importance

data = DataPreprocess.ReadData()

X_train, y_train, X_test, y_test, header = DataPreprocess.SplitData(data)

def DecisionTree(X_train, X_test):
    from sklearn.tree import DecisionTreeClassifier
    
    # fit the model
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    
    # get accuracy score
    predict = clf.predict(X_test)
    accuracy = accuracy_score(predict, y_test)
    
    importance = permutation_importance(clf, X_train, y_train, n_repeats=10)
    
    return accuracy, importance.importances_mean

def NaiveBayes(X_train, X_test):
    from sklearn.naive_bayes import GaussianNB
    
    # fit the model
    clf = GaussianNB()
    clf.fit(X_train, y_train)

    # get accuracy score
    predict = clf.predict(X_test)
    accuracy = accuracy_score(predict, y_test)

    importance = permutation_importance(clf, X_train, y_train, n_repeats=10)
    
    return accuracy, importance.importances_mean

def LogisticRegression(X_train, X_test):
    from sklearn.linear_model import LogisticRegression
    
    # fit the model
    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    # get accuracy score
    predict = clf.predict(X_test)
    accuracy = accuracy_score(predict, y_test)

    importance = permutation_importance(clf, X_train, y_train, n_repeats=10)

    return accuracy, importance.importances_mean

def RandomForest(X_train, X_test):
    from sklearn.ensemble import RandomForestClassifier
    
    # fit the model
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    
    # get accuract score
    predict = clf.predict(X_test)
    accuracy = accuracy_score(predict, y_test)
    
    importance = permutation_importance(clf, X_train, y_train, n_repeats=10)
    
    return accuracy, importance.importances_mean

def FindMinIndex(topN, importance, header):
    tempImportance = importance
    minHeader = []
    
    for i in range(topN):
        minIndex = np.argmin(tempImportance)
        minHeader.append(header[minIndex])   
        tempImportance[minIndex] = 10
        
    return minHeader

def Plot(header, importance, title):
    plt.figure()
    plt.xticks(rotation=90, ha='center')
    plt.title(title)
    plt.bar(header, importance)
    plt.savefig(title, bbox_inches='tight')
    
def DropColumns(dropCols):
    droppedData = data.drop(dropCols, axis = 1)
    dropped_X_train, y_train, dropped_X_test, y_test, droppedHeader = DataPreprocess.SplitData(droppedData)
    
    return dropped_X_train, dropped_X_test, droppedHeader

# normalize to 1-2
def Normalization(accuracy):
    return (accuracy - min(accuracy)) / (max(accuracy) - min(accuracy)) + 1


if __name__ == "__main__":
    print("Happy Machine Learning")

    #----------------------------------------------------------------------
    # Run with original full data
    
    origin_X_train, origin_y_train, origin_X_test, origin_y_test, origin_header = DataPreprocess.SplitDataOnly(data)
    originAccuracy = []
    
    # decision tree
    decisionTreeAccuracy, decisionTreeImportance = DecisionTree(origin_X_train, origin_X_test)
    print("Original Decision Tree Accuracy: ", decisionTreeAccuracy)
    originAccuracy.append(decisionTreeAccuracy)
    Plot(origin_header, decisionTreeImportance, "Original Decision Tree Importance")
    
    # random forest
    randomForestAccuracy, randomForestImportance = RandomForest(origin_X_train, origin_X_test)
    print("Original Random Forest Accuracy: ", randomForestAccuracy)
    originAccuracy.append(randomForestAccuracy)
    Plot(origin_header, randomForestImportance, "Original Random Forest Importance")

    # naive Bayes
    naiveBayesAccuracy, naiveBayesImportance = NaiveBayes(origin_X_train, origin_X_test)
    print("Original Naive Bayes Accuracy: ", naiveBayesAccuracy)
    originAccuracy.append(naiveBayesAccuracy)
    Plot(origin_header, naiveBayesImportance, "Original Naive Bayes")

    # LogisticRegression
    logisticRegressionAccuracy, logisticRegressionImportance = LogisticRegression(origin_X_train, origin_X_test)
    print("Original Logistic Regression Accuracy: ", logisticRegressionAccuracy)
    originAccuracy.append(logisticRegressionAccuracy)
    Plot(origin_header, logisticRegressionImportance, "Original Logistic Regression Importance")
    
    normalizedOriAcc = Normalization(originAccuracy)
    # combine the importances with normalized accuracy
    combinedImportance = normalizedOriAcc[0] * decisionTreeImportance + normalizedOriAcc[1] * randomForestImportance \
        + normalizedOriAcc[2] * naiveBayesImportance + normalizedOriAcc[3] * logisticRegressionImportance
    Plot(origin_header, combinedImportance, "Original Combined Importance")
    
    #----------------------------------------------------------------------
    # drop unnecessary data 
    
    initialAccuracy = []
    
    # decision tree
    decisionTreeAccuracy, decisionTreeImportance = DecisionTree(X_train, X_test)
    print("Decision Tree Accuracy: ", decisionTreeAccuracy)
    initialAccuracy.append(decisionTreeAccuracy)
    Plot(header, decisionTreeImportance, "Decision Tree Importance")
    
    # random forest
    randomForestAccuracy, randomForestImportance = RandomForest(X_train, X_test)
    print("Random Forest Accuracy: ", randomForestAccuracy)
    initialAccuracy.append(randomForestAccuracy)
    Plot(header, randomForestImportance, "Random Forest Importance")

    # naive Bayes
    naiveBayesAccuracy, naiveBayesImportance = NaiveBayes(X_train, X_test)
    print("Naive Bayes Accuracy: ", naiveBayesAccuracy)
    initialAccuracy.append(naiveBayesAccuracy)
    Plot(header, naiveBayesImportance, "Naive Bayes")

    # LogisticRegression
    logisticRegressionAccuracy, logisticRegressionImportance = LogisticRegression(X_train, X_test)
    print("Logistic Regression Accuracy: ", logisticRegressionAccuracy)
    initialAccuracy.append(logisticRegressionAccuracy)
    Plot(header, logisticRegressionImportance, "Logistic Regression Importance")
    
    normalizedIniAcc = Normalization(initialAccuracy)
    # combine the importances with normalized accuracy
    combinedImportance = normalizedIniAcc[0] * decisionTreeImportance + normalizedIniAcc[1] * randomForestImportance \
        + normalizedIniAcc[2] * naiveBayesImportance + normalizedIniAcc[3] * logisticRegressionImportance
    Plot(header, combinedImportance, "Dropped Combined Importance")
    
    #----------------------------------------------------------------------
    # Second round
    # drop the 10 least important feature
    dropHeader = FindMinIndex(10, combinedImportance, header)
    dropped_X_train, dropped_X_test, droppedHeader = DropColumns(dropHeader)
    
    droppedAccuracy = []
    
    # decision tree
    decisionTreeAccuracy, decisionTreeImportance = DecisionTree(dropped_X_train, dropped_X_test)
    print("Dropped Decision Tree Accuracy: ", decisionTreeAccuracy)
    droppedAccuracy.append(decisionTreeAccuracy)
    Plot(droppedHeader, decisionTreeImportance, "Dropped Decision Tree Importance")
    
    # random forest
    randomForestAccuracy, randomForestImportance = RandomForest(dropped_X_train, dropped_X_test)
    print("Dropped Random Forest Accuracy: ", randomForestAccuracy)
    droppedAccuracy.append(randomForestAccuracy)
    Plot(droppedHeader, randomForestImportance, "Dropped Random Forest Importance")

    # naive Bayes
    naiveBayesAccuracy, naiveBayesImportance = NaiveBayes(dropped_X_train, dropped_X_test)
    print("Dropped Naive Bayes Accuracy: ", naiveBayesAccuracy)
    droppedAccuracy.append(naiveBayesAccuracy)
    Plot(droppedHeader, naiveBayesImportance, "Dropped Naive Bayes")

    # LogisticRegression
    logisticRegressionAccuracy, logisticRegressionImportance = LogisticRegression(dropped_X_train, dropped_X_test)
    print("Dropped Logistic Regression Accuracy: ", logisticRegressionAccuracy)
    droppedAccuracy.append(logisticRegressionAccuracy)
    Plot(droppedHeader, logisticRegressionImportance, "Dropped Logistic Regression Importance")
    
    normalizedDropAcc = Normalization(droppedAccuracy)
    # combine the importances with normalized accuracy
    combinedImportance = normalizedDropAcc[0] * decisionTreeImportance + normalizedDropAcc[1] * randomForestImportance \
        + normalizedDropAcc[2] * naiveBayesImportance + normalizedDropAcc[3] * logisticRegressionImportance
    Plot(droppedHeader, combinedImportance, "Dropped Combined Importance")
    
    #----------------------------------------------------------------------
    # Third round
    # drop another 6 least important feature
    dropHeader2 = FindMinIndex(6, combinedImportance, droppedHeader)
    dropHeader2 = dropHeader + dropHeader2
    dropped2_X_train, dropped2_X_test, droppedHeader2 = DropColumns(dropHeader2)
    
    droppedAccuracy2 = []
    
    # decision tree
    decisionTreeAccuracy, decisionTreeImportance = DecisionTree(dropped2_X_train, dropped2_X_test)
    print("2nd Round Dropped Decision Tree Accuracy: ", decisionTreeAccuracy)
    droppedAccuracy2.append(decisionTreeAccuracy)
    Plot(droppedHeader2, decisionTreeImportance, "Dropped Decision Tree Importance 2")
    
    # random forest
    randomForestAccuracy, randomForestImportance = RandomForest(dropped2_X_train, dropped2_X_test)
    print("2nd Round Dropped Random Forest Accuracy: ", randomForestAccuracy)
    droppedAccuracy2.append(randomForestAccuracy)
    Plot(droppedHeader2, randomForestImportance, "Dropped Random Forest Importance 2")

    # naive Bayes
    naiveBayesAccuracy, naiveBayesImportance = NaiveBayes(dropped2_X_train, dropped2_X_test)
    print("2nd Round Dropped Naive Bayes Accuracy: ", naiveBayesAccuracy)
    droppedAccuracy2.append(naiveBayesAccuracy)
    Plot(droppedHeader2, naiveBayesImportance, "Dropped Naive Bayes 2")

    # LogisticRegression
    logisticRegressionAccuracy, logisticRegressionImportance = LogisticRegression(dropped2_X_train, dropped2_X_test)
    print("2nd Round Dropped Logistic Regression Accuracy: ", logisticRegressionAccuracy)
    droppedAccuracy2.append(logisticRegressionAccuracy)
    Plot(droppedHeader2, logisticRegressionImportance, "Dropped Logistic Regression Importance 2")
    
    normalizedDrop2Acc = Normalization(droppedAccuracy2)
    # combine the importances with normalized accuracy
    combinedImportance = normalizedDrop2Acc[0] * decisionTreeImportance + normalizedDrop2Acc[1] * randomForestImportance \
        + normalizedDrop2Acc[2] * naiveBayesImportance + normalizedDrop2Acc[3] * logisticRegressionImportance
    Plot(droppedHeader2, combinedImportance, "Dropped Combined Importance 2")
    
    #----------------------------------------------------------------------
    # plot accuracy
    xAxis = ["Decision Tree", "Random Forest", "Naive Bayes", "Logistic Regression"]
    plt.figure()
    plt.xticks(rotation=90, ha='center')
    plt.title("Accuracy")
    plt.plot(xAxis, originAccuracy, 'v', color='black', label='Original')
    plt.plot(xAxis, initialAccuracy, 'v', color='blue', label='Initial')
    plt.plot(xAxis, droppedAccuracy, 'v', color='red', label='Drop 10')
    plt.plot(xAxis, droppedAccuracy2, 'v', color='green', label='Drop 16')
    plt.legend()
    
    plt.savefig("Accuracy", bbox_inches='tight')
    
    
    
    
# %%
