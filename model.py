import numpy as np# id:5--5--5-0 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_curve

dfi = pd.read_csv("week4i.csv")
X1i = dfi.iloc[:,0]
X2i = dfi.iloc[:,1]
Xi = np.column_stack((X1i,X2i))
yi = dfi.iloc[:,2]

dfii = pd.read_csv("week4ii.csv")
X1ii = dfii.iloc[:,0]
X2ii = dfii.iloc[:,1]
Xii = np.column_stack((X1ii,X2ii))
yii = dfii.iloc[:,2]

def trainingDataScatterPlot(features, output):
    plt.figure()
    plt.scatter(features[output > 0, 0], features[output > 0, 1], color="red")
    plt.scatter(features[output < 0, 0], features[output < 0, 1], color="green")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend(["Positive Output","Negative Output"])
    plt.show()

def predictionsScatterPlot(X, yPredictions, title):
    plt.figure()
    plt.scatter(X[yPredictions > 0, 0], X[yPredictions > 0, 1], color="red")
    plt.scatter(X[yPredictions < 0, 0], X[yPredictions < 0, 1], color="green") 
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2") 
    plt.title(title)
    plt.legend(["Positive Output","Negative Output"])    
    plt.show()

def crossValPlot(CValues, prec_mean_err, prec_std_err, f1_mean_err, f1_std_err,  title):
    plt.figure()
    plt.errorbar(CValues, prec_mean_err, label='Precision Score', ecolor='red', color='blue', yerr=prec_std_err, linewidth=2)
    plt.errorbar(CValues, f1_mean_err, label='F1 Score', ecolor='g', color='orange', yerr=f1_std_err, linewidth=2)
    plt.xlabel("C")
    plt.xlim(0,1000)
    # plt.xlim(0,100)
    # plt.xlim(0,10)
    plt.ylabel("F1 Accuracy & Precision")
    plt.legend()
    plt.title(title)
    plt.show()

def KNNPlot(KValues, prec_mean_err, prec_std_err, f1_mean_err, f1_std_err,  title):
    plt.figure()
    plt.errorbar(KValues, prec_mean_err, label='Precision Score', ecolor='red', color='blue', yerr=prec_std_err, linewidth=2)
    plt.errorbar(KValues, f1_mean_err, label='F1 Score', ecolor='g', color='orange', yerr=f1_std_err, linewidth=2)
    plt.xlabel("K Value")
    plt.xlim(1,10)
    plt.ylabel("F1 Accuracy & Precision")
    plt.legend()
    plt.title(title)
    plt.show()

def crossValLReg(features, output):
    # Ci = [1, 2, 4, 6, 8, 10]
    Ci = [1, 10, 50, 100, 1000]
    # Ci = [100]
    for polyFeatures in [2, 3, 4, 5, 6]:
        prec_mean_err = []; prec_std_err = []
        f1_mean_err = []; f1_std_err = []
        poly = PolynomialFeatures(polyFeatures)
        result = poly.fit_transform(features)
        for C in Ci:
            model = LogisticRegression(max_iter=1000, C=C, penalty="l2")
            model.fit(result, output)
            predictions = model.predict(result)
            title = "Polynomial: " + str(polyFeatures) + ", C: " + str(C)
            predictionsScatterPlot(features, predictions, title)
            # Cross Validation K-Fold
            f1Scores = cross_val_score(model, result, output, scoring='f1', cv=5) # f1 scores
            precScores = cross_val_score(model, result, output, scoring='precision', cv=5) # precision scores
            prec_std_err.append(np.array(precScores).std())
            prec_mean_err.append(np.array(precScores).mean())
            f1_std_err.append(np.array(f1Scores).std())
            f1_mean_err.append(np.array(f1Scores).mean())
            tn, fp, fn, tp = confusion_matrix(output, predictions).ravel()
            accuracy = (tn + tp)/(tn + tp + fn + fp)
            truePosRate = (tp)/(tp + fn)
            falsePosRate = (fp)/(tn + fp)
            precision = (tp)/(tp + fp)
            print("Accuracy: \n", accuracy)
            print("True Positive Rate: \n", truePosRate)
            print("False Positive Rate: \n", falsePosRate)
            print("Precision: \n", precision)
            ConfusionMatrixDisplay.from_predictions(output, predictions)
            plt.title("Polynomial: " + str(polyFeatures) + ", C: " + str(C) + " Confusion Matrix")
            plt.show()
        crossValPlot(Ci, prec_mean_err, prec_std_err, f1_mean_err, f1_std_err, "Cross Validation Precision & F1 Accuracy: P=" + str(polyFeatures))

def crossValKNN(features, output):
    Ki = [3, 4, 5, 6, 7, 8, 9, 10]
    prec_mean_err = []; prec_std_err = []
    f1_mean_err = []; f1_std_err = []    
    for K in Ki:
        model = KNeighborsClassifier(n_neighbors=K, weights='uniform')
        model.fit(features, output)
        predictions = model.predict(features)
        title = "KNN Predictions, K: " + str(K)
        predictionsScatterPlot(features, predictions, title)
        f1Scores = cross_val_score(model, features, output, scoring='f1', cv=5) # f1 scores
        precScores = cross_val_score(model, features, output, scoring='precision', cv=5) # precision scores
        prec_std_err.append(np.array(precScores).std())
        prec_mean_err.append(np.array(precScores).mean())
        f1_std_err.append(np.array(f1Scores).std())
        f1_mean_err.append(np.array(f1Scores).mean())
        tn, fp, fn, tp = confusion_matrix(output, predictions).ravel()
        accuracy = (tn + tp)/(tn + tp + fn + fp)
        truePosRate = (tp)/(tp + fn)
        falsePosRate = (fp)/(tn + fp)
        precision = (tp)/(tp + fp)
        print("Accuracy: \n", accuracy)
        print("True Positive Rate: \n", truePosRate)
        print("False Positive Rate: \n", falsePosRate)
        print("Precision: \n", precision)
        ConfusionMatrixDisplay.from_predictions(output, predictions)
        plt.title("K: " + str(K) + " Confusion Matrix")
        plt.show()
    KNNPlot(Ki, prec_mean_err, prec_std_err, f1_mean_err, f1_std_err, "KNN Scores")

def randClassifier(features, output):
    dummy_clf = DummyClassifier(strategy="uniform")
    dummy_clf.fit(features, output)
    predictions = dummy_clf.predict(features)
    tn, fp, fn, tp = confusion_matrix(output, predictions).ravel()
    accuracy = (tn + tp)/(tn + tp + fn + fp)
    truePosRate = (tp)/(tp + fn)
    falsePosRate = (fp)/(tn + fp)
    precision = (tp)/(tp + fp)
    print("Accuracy: \n", accuracy)
    print("True Positive Rate: \n", truePosRate)
    print("False Positive Rate: \n", falsePosRate)
    print("Precision: \n", precision)
    ConfusionMatrixDisplay.from_predictions(output, predictions)
    plt.title("Random Classifier Confusion Matrix")
    plt.show()

def ROCPlot(features, output):
    # poly = PolynomialFeatures(2) # P = 2 optimum
    poly = PolynomialFeatures(8) # P = 8 optimum
    result = poly.fit_transform(features)
    model = LogisticRegression(max_iter=1000, C=4, penalty="l2") # C = 4 optimum
    model.fit(result, output)
    LRegPredictions = model.predict(result)
    fprLReg, tprLReg, _LReg = roc_curve(output, LRegPredictions)

    # model = KNeighborsClassifier(n_neighbors=4, weights='uniform') # K = 4 optimum
    model = KNeighborsClassifier(n_neighbors=5, weights='uniform') # K = 5 optimum
    model.fit(features, output)
    KnnPredictions = model.predict(features)
    fprKnn, tprKnn, _Knn = roc_curve(output, KnnPredictions)

    dummy_clf = DummyClassifier(strategy="uniform")
    dummy_clf.fit(features, output)
    randPredictions = dummy_clf.predict(features)
    fprRand, tprRand, _Rand = roc_curve(output, randPredictions)

    plt.plot(fprLReg, tprLReg, color='g')
    plt.plot(fprKnn, tprKnn, color='b')
    plt.plot(fprRand, tprRand, color='orange', linestyle='--')
    plt.xlabel("False positive Rate")
    plt.ylabel("True positive Rate")
    plt.title("ROC curves")
    plt.legend(["Logistic Regression", "K - Nearest Neighbors", "Random Classifier"])
    plt.show()

# trainingDataScatterPlot(Xi, yi) # training data scatter
crossValLReg(Xi, yi)
# crossValKNN(Xi, yi)
# randClassifier(Xi, yi)
# ROCPlot(Xi, yi)

# trainingDataScatterPlot(Xii, yii) # training data scatter
# crossValLReg(Xii, yii)
# crossValKNN(Xii, yii)
# randClassifier(Xii, yii)
# ROCPlot(Xii, yii)
