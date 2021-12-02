from re import match
import numpy as np
from numpy.lib.function_base import average# id:5--5--5-0 
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
from sklearn import metrics
from sklearn import linear_model
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

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

def calulateMetrics(confusion_m): 
    tpA = confusion_m[0]
    tnA = confusion_m[4]+confusion_m[5]+confusion_m[7]+confusion_m[8]
    fpA = confusion_m[3]+confusion_m[6]
    fnA = confusion_m[1]+confusion_m[2]
    # print("tpA:", tpA, "tnA: ", tnA, "fpA:", fpA, "fnA: ", fnA)
    accuracyA = (tnA + tpA)/(tnA + fpA + tpA + fnA)
    precisionA = (tpA)/(tpA + fpA)
    recallA = (tpA)/(tpA + fnA)
    f1ScoreA = 2 * ((precisionA * recallA)/(precisionA + recallA))
    print("F1A: ", f1ScoreA)

    tpH = confusion_m[8]
    tnH = confusion_m[0]+confusion_m[1]+confusion_m[3]+confusion_m[4]
    fpH = confusion_m[2]+confusion_m[5]
    fnH = confusion_m[6]+confusion_m[7]
    # print("tpH:", tpH, "tnH: ", tnH, "fpH:", fpH, "fnH: ", fnH)
    accuracyH = (tnH + tpH)/(tnH + fpH + tpH + fnH)
    precisionH = (tpH)/(tpH + fpH)
    recallH = (tpH)/(tpH + fnH)
    if (precisionH == 0 or recallH == 0): f1ScoreH = 0
    else: f1ScoreH = 2 * ((precisionH * recallH)/(precisionH + recallH))
    print("F1H: ", f1ScoreH)

    tpD = confusion_m[4]
    tnD = confusion_m[0]+confusion_m[2]+confusion_m[6]+confusion_m[8]
    fpD = confusion_m[1]+confusion_m[7]
    fnD = confusion_m[3]+confusion_m[5]
    # print("tpD:", tpD, "tnD: ", tnD, "fpD:", fpD, "fnD: ", fnD)
    accuracyD = (tnD + tpD)/(tnD + fpD + tpD + fnD)
    precisionD = (tpD)/(tpD + fpD)
    recallD = (tpD)/(tpD + fnD)
    if (precisionD == 0 or recallD == 0): f1ScoreD = 0
    else: f1ScoreD = 2 * ((precisionD * recallD)/(precisionD + recallD))
    print("F1D: ", f1ScoreD)

    f1ScoreTotal = (f1ScoreA + f1ScoreH + f1ScoreD) / 3
    return f1ScoreTotal
    # Return metric once we decide what to use
    # return accuracyD, precisionD, accuracyA, precisionA ... etc
    
def printPerformance(probabilities, predictions, output, rowIndex, C, f1_scores, title):
    print(title+" for C: "+str(C)) 
    # Alphabetical order left to right
    #print("Probabilities: ", probabilities)
    #print("Predicted result: ",predictions)
    #print("Actual result:    ",np.array(output[rowIndex-10:rowIndex+1]))
    # print("ROC: ", roc_auc_score(output[rowIndex-10:rowIndex+1], predictions,  multi_class='ovr'))
    print("\n")
    confusion_m = confusion_matrix(np.array(output[rowIndex-10:rowIndex+1]), predictions).ravel()
    print("Confusion Matrix", confusion_m)
    if(len(confusion_m)==9): # Gameweek 17 had no Draws so confusion matrix is 2x2 (need to handle this)
        newF1 = calulateMetrics(confusion_m)
        f1_scores.append(newF1)

    # print(metrics.classification_report(np.array(output[rowIndex-10:rowIndex+1]), predictions, digits=3, zero_division=0))
    # ConfusionMatrixDisplay.from_predictions(np.array(output[rowIndex-10:rowIndex+1]), predictions)
    # title = "Confusion Matrix " + title + " Gameweek " + str(gameweek)
    # plt.title(title)
    # plt.show()

def printLassoRidgePerformance(linearOutput, predictions, output, rowIndex, C, f1_scores, title):
    predictions[predictions<=-.33] = -1
    predictions[predictions>=.33] = 1
    predictions[abs(predictions)<1] = 0

    print(title+" for C: "+str(C)) 
    # Alphabetical order left to right
    #print("Predicted result: ",predictions)
    #print("Actual result:    ",np.array(output[rowIndex-10:rowIndex+1]))

    result = np.zeros((10, 1))
    for i in range(10):
        result[i] = linearOutput[i]
    # print("ROC: ", roc_auc_score(output[rowIndex-10:rowIndex+1], predictions,  multi_class='ovr'))
    confusion_m = confusion_matrix(result, predictions).ravel()
    print("Confusion Matrix", confusion_m)
    if(len(confusion_m)==9): # Gameweek 17 had no Draws so confusion matrix is 2x2 (need to handle this)
        newF1 = calulateMetrics(confusion_m)
        f1_scores.append(newF1)
    #print("\n")
    # print(metrics.classification_report(np.array(output[rowIndex-10:rowIndex+1]), predictions, digits=3, zero_division=0))
    # ConfusionMatrixDisplay.from_predictions(np.array(output[rowIndex-10:rowIndex+1]), predictions)
    # title = "Confusion Matrix " + title + " Gameweek " + str(gameweek)
    # plt.title(title)
    # plt.show()

def LogisticReg(features, output, rowIndex): 
    Ci_range = [ 
        0.001, .01, .1, 1, 10, 1000
    ]
    f1_scores = []
    for C in Ci_range:
        model = LogisticRegression(max_iter=1000, C = C, penalty="l2")
        model.fit(features[0:rowIndex-10], output[0:rowIndex-10])
        predictions = model.predict(np.array(features[rowIndex-10:rowIndex+1]))
        probabilities = model.predict_proba(np.array(features[rowIndex-10:rowIndex+1]))
        printPerformance(probabilities, predictions, output, rowIndex, C, f1_scores, "* LogisticRegression *")
    print("SCORES: ", f1_scores)
    plotCrossValidation(Ci_range, f1_scores, 'C Values', 'F1 Score', "* logisticReg *")

def knn(features, output, rowIndex): 
    Ki_range = [ 
        2, 3, 4, 5, 6
    ]
    f1_scores = []
    for n_neighbours in Ki_range:
        model = KNeighborsClassifier(n_neighbours, weights='uniform')
        model.fit(features[0:rowIndex-10], output[0:rowIndex-10])
        probabilities = model.predict_proba(np.array(features[rowIndex-10:rowIndex+1]))
        predictions = model.predict(np.array(features[rowIndex-10:rowIndex+1]))
        printPerformance(probabilities, predictions, output, rowIndex, n_neighbours, f1_scores, "* KNeighborsClassifier *")
    print("SCORES: ", f1_scores)
    plotCrossValidation(Ki_range, f1_scores, 'C Values', 'F1 Score', "* KNN *")

def randomClassifier(features, output, rowIndex): 
    dummy_clf = DummyClassifier(strategy="uniform")
    dummy_clf.fit(features[0:rowIndex-10], output[0:rowIndex-10])
    probabilities = dummy_clf.predict_proba(np.array(features[rowIndex-10:rowIndex+1]))
    predictions = dummy_clf.predict(np.array(features[rowIndex-10:rowIndex+1]))
    printPerformance(probabilities, predictions, output, rowIndex, "* RandomClassifier *")

def KFlassoReg(features, output, rowIndex): 
    C=1
    linearOutput = np.where(output == 'A', -1.0, output)
    linearOutput = np.where(linearOutput == 'H', 1.0, linearOutput)
    linearOutput = np.where(linearOutput == 'D', 0.0, linearOutput)
    Ci_range = [ 
        0.5, 1, 5, 10, 50, 100
    ]
    mean_error=[]; std_error=[]
    for C in Ci_range:
        model = linear_model.Lasso(alpha=1/(2*C), max_iter=2000)# warning said to increase number of iterations
        temp = []
        kf = KFold(n_splits=5)
        for train, test in kf.split(features[0:rowIndex-10]):
            model.fit(features[train], linearOutput[train])
            predictions = model.predict(np.array(features[test]))
            # printLassoRidgePerformance(linearOutput[test], predictions[test], output, rowIndex, C, "* lassoReg *")
            temp.append(mean_squared_error(linearOutput[test],predictions))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
    print("Mean Error = ", mean_error)
    print("Standard Deviation Error = ", std_error)

def lassoReg(features, output, rowIndex): 
    #C=1
    linearOutput = np.where(output == 'A', -1.0, output)
    linearOutput = np.where(linearOutput == 'H', 1.0, linearOutput)
    linearOutput = np.where(linearOutput == 'D', 0.0, linearOutput)
    Ci_range = [ 
        0.5, 1, 5, 10, 50, 100
    ]
    f1_scores = []
    for C in Ci_range:
        model = linear_model.Lasso(alpha=1/(2*C))
        model.fit(features[0:rowIndex-10], linearOutput[0:rowIndex-10])
        predictions = model.predict(np.array(features[rowIndex-10:rowIndex+1]))
        printLassoRidgePerformance(linearOutput, predictions, output, rowIndex, C, f1_scores, "* lassoReg *")

    print("SCORES: ", f1_scores)
    plotCrossValidation(Ci_range, f1_scores, 'C Values', 'F1 Score', "* lassoReg *")

def ridgeReg(features, output, rowIndex): 
    #C=1
    # Try different values for C
    linearOutput = np.where(output == 'A', -1, output)
    linearOutput = np.where(linearOutput == 'H', 1, linearOutput)
    linearOutput = np.where(linearOutput == 'D', 0, linearOutput)
    Ci_range = [ 
        0.5, 1, 5, 10, 50, 100
    ]
    f1_scores = []
    for C in Ci_range:
        model = linear_model.Ridge(alpha=1/(2*C))
        model.fit(features[0:rowIndex-10], linearOutput[0:rowIndex-10])
        predictions = model.predict(np.array(features[rowIndex-10:rowIndex+1]))
        printLassoRidgePerformance(linearOutput, predictions, output, rowIndex, C, f1_scores, "* ridgeReg *")

    print("SCORES: ", f1_scores)
    plotCrossValidation(Ci_range, f1_scores, 'C Values', 'F1 Score', "* ridgeReg *")

def plotCrossValidation(Clist, f1Score, xAxisTitle, yAxisTitle, title):
    plt.figure()
    plt.rc('font', size=18)
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.plot(Clist, f1Score, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
    plt.xlabel(xAxisTitle); 
    plt.ylabel(yAxisTitle); 
    plt.title(title)
    plt.show()

def modelTraining(features, output, rowIndex):    
    #  LogisticReg(features, output, rowIndex)
    # knn(features, output, rowIndex)
    # randomClassifier(features, output, rowIndex)
    # lassoReg(features, output, rowIndex)
    # ridgeReg(features, output, rowIndex)
    KFlassoReg(features, output, rowIndex)

rowIndex = 331
gameweek = 33
while gameweek < 39: #380 matches played by 20 teams 
    rowIndex+=10
    print("Gameweek: ", gameweek) 
    gameweekFeatures = np.column_stack((attendance[0:rowIndex],HomeWinStreak[0:rowIndex],HomeTotalPoints[0:rowIndex],HomeTotalGoalsScored[0:rowIndex],HomeTotalGoalsConceded[0:rowIndex],HomePreviousSeasonPoints[0:rowIndex],HomeTeamValue[0:rowIndex],AwayWinStreak[0:rowIndex],AwayTotalPoints[0:rowIndex],AwayTotalGoalsScored[0:rowIndex],AwayTotalGoalsConceded[0:rowIndex],AwayPreviousSeasonPoints[0:rowIndex],AwayTeamValue[0:rowIndex]))
    modelTraining(gameweekFeatures, matchResult[0:rowIndex], rowIndex)
    gameweek+=1