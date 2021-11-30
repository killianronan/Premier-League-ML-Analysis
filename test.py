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

scores1819 = pd.read_csv("cleaned201819scores.csv")
weekNumber = scores1819.iloc[:,1]
attendance = scores1819.iloc[:,2]
FTHomeGoals = scores1819.iloc[:,3]
HomeTeam = scores1819.iloc[:,4]
HomeWinStreak = scores1819.iloc[:,5]
HomeTotalWins = scores1819.iloc[:,6]
HomeTotalPoints = scores1819.iloc[:,7]
HomeTotalGoalsScored = scores1819.iloc[:,8]
HomeTotalGoalsConceded = scores1819.iloc[:,9] # Maybe do a total for the past 5 games to get a more accurate estimate
HomePreviousSeasonPoints = scores1819.iloc[:,10]
HomeTeamValue = scores1819.iloc[:,11]
FTAwayGoals = scores1819.iloc[:,12]
AwayTeam = scores1819.iloc[:,13]
AwayWinStreak = scores1819.iloc[:,14]
AwayTotalWins = scores1819.iloc[:,15]
AwayTotalPoints = scores1819.iloc[:,16]
AwayTotalGoalsScored = scores1819.iloc[:,17]
AwayTotalGoalsConceded = scores1819.iloc[:,18]
AwayPreviousSeasonPoints = scores1819.iloc[:,19]
AwayTeamValue = scores1819.iloc[:,20]
matchResult = scores1819.iloc[:,21]

features1819 = np.column_stack((attendance,HomeWinStreak,HomeTotalPoints,HomeTotalGoalsScored,HomeTotalGoalsConceded,HomePreviousSeasonPoints,HomeTeamValue,AwayWinStreak,AwayTotalPoints,AwayTotalGoalsScored,AwayTotalGoalsConceded,AwayPreviousSeasonPoints,AwayTeamValue)) 
print(features1819)
output = matchResult

def predictionsScatterPlot(X, yPredictions, title):
    plt.figure()
    plt.scatter(X[yPredictions > 0, 0], X[yPredictions > 0, 1], color="red")
    plt.scatter(X[yPredictions < 0, 0], X[yPredictions < 0, 1], color="green") 
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2") 
    plt.title(title)
    plt.legend(["Positive Output","Negative Output"])    
    plt.show()

def modelTraining(features, output):
    Ci = [1, 10, 50, 100, 1000]
    prec_mean_err = []; prec_std_err = []
    f1_mean_err = []; f1_std_err = []
    for C in Ci:
        model = LogisticRegression(max_iter=1000, C=C, penalty="l2")
        model.fit(features, output)
        predictions = model.predict(features)
        predictionsScatterPlot(features, predictions, "Predictions")
        # Cross Validation K-Fold
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
        plt.title("Confusion Matrix")
        plt.show()
    # crossValPlot(Ci, prec_mean_err, prec_std_err, f1_mean_err, f1_std_err, "Cross Validation Precision & F1 Accuracy: P=" + str(polyFeatures))

modelTraining(features1819, matchResult)
