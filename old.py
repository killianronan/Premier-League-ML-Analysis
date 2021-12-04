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
    if((tnA+tpA)!=0 and (tnA + fpA + tpA + fnA)!=0):
        accuracyA = (tnA + tpA)/(tnA + fpA + tpA + fnA)
    else: accuracyA = 0
    if(tpA!=0 and (tpA + fpA)!=0):
        precisionA = (tpA)/(tpA + fpA)
    else: precisionA = 0
    if(tpA!=0 and (tpA + fnA)!=0):
        recallA = (tpA)/(tpA + fnA)
    else: recallA = 0

    if(precisionA != 0 and recallA != 0):
        f1ScoreA = 2 * ((precisionA * recallA)/(precisionA + recallA))
    else: f1ScoreA = 0
    # print("F1A: ", f1ScoreA)

    tpH = confusion_m[8]
    tnH = confusion_m[0]+confusion_m[1]+confusion_m[3]+confusion_m[4]
    fpH = confusion_m[2]+confusion_m[5]
    fnH = confusion_m[6]+confusion_m[7]
    # print("tpH:", tpH, "tnH: ", tnH, "fpH:", fpH, "fnH: ", fnH)
    if((tnH+tpH)!=0 and (tnH + fpH + tpH + fnH)!=0):
        accuracyH = (tnH + tpH)/(tnH + fpH + tpH + fnH)
    else: accuracyH = 0
    if(tpH!=0 and (tpH + fpH)!=0):
        precisionH = (tpH)/(tpH + fpH)
    else: precisionH = 0
    if(tpH!=0 and (tpH + fnH)!=0):
        recallH = (tpH)/(tpH + fnH)
    else: recallH = 0

    if(precisionH != 0 and recallH != 0):
        f1ScoreH = 2 * ((precisionH * recallH)/(precisionH + recallH))
    else: f1ScoreH = 0
    # print("F1H: ", f1ScoreH)

    tpD = confusion_m[4]
    tnD = confusion_m[0]+confusion_m[2]+confusion_m[6]+confusion_m[8]
    fpD = confusion_m[1]+confusion_m[7]
    fnD = confusion_m[3]+confusion_m[5]
    # print("tpD:", tpD, "tnD: ", tnD, "fpD:", fpD, "fnD: ", fnD)
    if((tnD+tpD)!=0 and (tnD + fpD + tpD + fnD)!=0):
        accuracyD = (tnD + tpD)/(tnD + fpD + tpD + fnD)
    else: accuracyD = 0
    if(tpD!=0 and (tpD + fpD)!=0):
        precisionD = (tpD)/(tpD + fpD)
    else: precisionD = 0
    if(tpD!=0 and (tpD + fnD)!=0):
        recallD = (tpD)/(tpD + fnD)
    else: recallD = 0

    if(precisionD != 0 and recallD != 0):
        f1ScoreD = 2 * ((precisionD * recallD)/(precisionD + recallD))
    else: f1ScoreD = 0
    # print("F1D: ", f1ScoreD)

    if((f1ScoreA + f1ScoreH + f1ScoreD) != 0):
        f1ScoreTotal = (f1ScoreA + f1ScoreH + f1ScoreD) / 3
    else: f1ScoreTotal = 0

    return f1ScoreTotal
    # return accuracyD, precisionD, accuracyA, precisionA ... etc
    
def printPerformance(probabilities, linearOutput, predictions, output, rowIndex, C, f1_scores, title):
    # print(title+" for C: "+str(C)) 
    # Alphabetical order left to right
    #print("Probabilities: ", probabilities)
    #print("Predicted result: ",predictions)
    #print("Actual result:    ",np.array(output[rowIndex-10:rowIndex+1]))
    # print("ROC: ", roc_auc_score(output[rowIndex-10:rowIndex+1], predictions,  multi_class='ovr'))
    if (probabilities == 0): confusion_m = confusion_matrix(linearOutput, predictions).ravel()
    else: confusion_m = confusion_matrix(np.array(output[rowIndex-10:rowIndex+1]), predictions).ravel()
    # print("Confusion Matrix", confusion_m)
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

    # print(title+" for C: "+str(C)) 
    # Alphabetical order left to right
    #print("Predicted result: ",predictions)
    #print("Actual result:    ",np.array(output[rowIndex-10:rowIndex+1]))

    result = np.zeros((len(predictions), 1))
    for i in range(len(predictions)):
        result[i] = linearOutput[i]
    # print("ROC: ", roc_auc_score(output[rowIndex-10:rowIndex+1], predictions,  multi_class='ovr'))
    print("result: ", result)
    print("pred: ", predictions)
    confusion_m = confusion_matrix(result, predictions).ravel()
    # print("Confusion Matrix", confusion_m)
    if(len(confusion_m)==9): # Gameweek 17 had no Draws so confusion matrix is 2x2 (need to handle this)
        newF1 = calulateMetrics(confusion_m)
        f1_scores.append(newF1)
    #print("\n")
    # print(metrics.classification_report(np.array(output[rowIndex-10:rowIndex+1]), predictions, digits=3, zero_division=0))
    # ConfusionMatrixDisplay.from_predictions(np.array(output[rowIndex-10:rowIndex+1]), predictions)
    # title = "Confusion Matrix " + title + " Gameweek " + str(gameweek)
    # plt.title(title)
    # plt.show()

def convertCharToNumber(actual, predictions):
    actualOutput = np.where(actual == 'A', -1.0, actual)
    actualOutput = np.where(actualOutput == 'H', 1.0, actualOutput)
    actualOutput = np.where(actualOutput == 'D', 0.0, actualOutput)
    output = np.where(predictions == 'A', -1.0, predictions)
    output = np.where(output == 'H', 1.0, output)
    output = np.where(output == 'D', 0.0, output)
    return actualOutput, output

def KLogisticReg(features, output, rowIndex): 
    Ci_range = [ 
        0.001, .01, .1, 1, 10, 100
    ]
    mean_error=[]; std_error=[]; f1_mean = []; f1_std_dev = []
    for C in Ci_range:
        model = LogisticRegression(max_iter=1000, C = C, penalty="l2")
        temp = []; f1_scores = []
        kf = KFold(n_splits=5)
        for train, test in kf.split(features[0:rowIndex-10]):
            model.fit(features[train], output[train])
            #if(rowIndex == 21 or rowIndex == 151 or rowIndex == 251 or rowIndex == 351): plotWeights(model.coef_ , 'K-Fold Logistic Regression', rowIndex)
            predictions = model.predict(np.array(features[test]))
            printPerformance(0, output[test], predictions, output, rowIndex, C, f1_scores, "* LogisticReg *")
            convertedActual, convertedPredictions = convertCharToNumber(output,predictions)
            temp.append(mean_squared_error(convertedActual[test],convertedPredictions))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
        f1_mean.append(np.array(f1_scores).mean())
        f1_std_dev.append(np.array(f1_scores).std())
    drawGraphs(Ci_range, mean_error, std_error, f1_mean, f1_std_dev, "LogisticReg")

def LogisticReg(features, output, rowIndex): 
    Ci_range = [ 
        0.001, .01, .1, 1, 10, 1000
    ]
    f1_scores = []
    for C in Ci_range:
        model = LogisticRegression(max_iter=1000, C = C, penalty="l2")
        model.fit(features[0:rowIndex-10], output[0:rowIndex-10])
        if(rowIndex == 21 or rowIndex == 151 or rowIndex == 251 or rowIndex == 351): plotWeights(model.coef_ , 'Logistic Regression', rowIndex)  ## NEED TO EITHER PICK A C VALUE TO STICK WITH OR PASS C VALUE THROUGH TO THIS FUNCTION
        predictions = model.predict(np.array(features[rowIndex-10:rowIndex+1]))
        probabilities = model.predict_proba(np.array(features[rowIndex-10:rowIndex+1]))
        printPerformance(probabilities, predictions, output, rowIndex, C, f1_scores, "* LogisticRegression *")
    print("SCORES: ", f1_scores)
    plotF1Score(Ci_range, f1_scores, 'C Values', 'F1 Score', "* logisticReg *")

def Kknn(features, output, rowIndex): 
    Ki_range = [ 
        2, 3, 4, 5, 6
    ]
    mean_error=[]; std_error=[]; f1_mean = []; f1_std_dev = []
    for n_neighbours in Ki_range:
        model = KNeighborsClassifier(n_neighbours, weights='uniform')
        temp = []; f1_scores = []
        kf = KFold(n_splits=5)
        for train, test in kf.split(features[0:rowIndex-10]):
            model.fit(features[train], output[train])
            #if(rowIndex == 21 or rowIndex == 151 or rowIndex == 251 or rowIndex == 351): plotWeights(model.coef_ , 'KNN', rowIndex) #Need to fix this one up a bit will do tomorrow
            predictions = model.predict(np.array(features[test]))
            printPerformance(0, output[test], predictions, output, rowIndex, n_neighbours, f1_scores, "* KNN *")
            convertedActual, convertedPredictions = convertCharToNumber(output,predictions)
            temp.append(mean_squared_error(convertedActual[test],convertedPredictions))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
        f1_mean.append(np.array(f1_scores).mean())
        f1_std_dev.append(np.array(f1_scores).std())
    drawGraphs(Ki_range, mean_error, std_error, f1_mean, f1_std_dev, "KNN")

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
    plotF1Score(Ki_range, f1_scores, 'C Values', 'F1 Score', "* KNN *")

def randomClassifier(features, output, rowIndex): 
    dummy_clf = DummyClassifier(strategy="uniform")
    dummy_clf.fit(features[0:rowIndex-10], output[0:rowIndex-10])
    probabilities = dummy_clf.predict_proba(np.array(features[rowIndex-10:rowIndex+1]))
    predictions = dummy_clf.predict(np.array(features[rowIndex-10:rowIndex+1]))
    printPerformance(probabilities, predictions, output, rowIndex, "* RandomClassifier *")

def KLassoReg(features, output, rowIndex): 
    linearOutput = np.where(output == 'A', -1.0, output)
    linearOutput = np.where(linearOutput == 'H', 1.0, linearOutput)
    linearOutput = np.where(linearOutput == 'D', 0.0, linearOutput)
    Ci_range = [ 
        0.5, 1, 5, 10, 50, 100
    ]
    mean_error=[]; std_error=[]; f1_mean = []; f1_std_dev = []
    for C in Ci_range:
        model = linear_model.Lasso(alpha=1/(2*C), max_iter=2000)# warning said to increase number of iterations
        temp = []; f1_scores = []
        kf = KFold(n_splits=5)
        for train, test in kf.split(features[0:rowIndex-10]):
            model.fit(features[train], linearOutput[train])
            if(rowIndex == 21 or rowIndex == 151 or rowIndex == 251 or rowIndex == 351): plotWeights(model.coef_ , 'K-Lasso Regression', rowIndex)
            predictions = model.predict(np.array(features[test]))
            printLassoRidgePerformance(linearOutput[test], predictions, output, rowIndex, C, f1_scores, "* lassoReg *")
            temp.append(mean_squared_error(linearOutput[test],predictions))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
        f1_mean.append(np.array(f1_scores).mean())
        f1_std_dev.append(np.array(f1_scores).std())
    drawGraphs(Ci_range, mean_error, std_error, f1_mean, f1_std_dev, "lassoReg")

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
        if(rowIndex == 21 or rowIndex == 151 or rowIndex == 251 or rowIndex == 351): plotWeights(model.coef_ , 'Lasso Regression', rowIndex)
        predictions = model.predict(np.array(features[rowIndex-10:rowIndex+1]))
        printLassoRidgePerformance(linearOutput, predictions, output, rowIndex, C, f1_scores, "* lassoReg *")

    print("SCORES: ", f1_scores)
    plotF1Score(Ci_range, f1_scores, 'C Values', 'F1 Score', "* lassoReg *")

def KRidgeReg(features, output, rowIndex): 
    linearOutput = np.where(output == 'A', -1.0, output)
    linearOutput = np.where(linearOutput == 'H', 1.0, linearOutput)
    linearOutput = np.where(linearOutput == 'D', 0.0, linearOutput)
    Ci_range = [ 
        0.5, 1, 5, 10, 50, 100
    ]
    mean_error=[]; std_error=[]; f1_mean = []; f1_std_dev = []
    for C in Ci_range:
        model = linear_model.Ridge(alpha=1/(2*C))# warning said to increase number of iterations
        temp = []; f1_scores = []
        kf = KFold(n_splits=5)
        for train, test in kf.split(features[0:rowIndex-10]):
            model.fit(features[train], linearOutput[train])
            if(rowIndex == 21 or rowIndex == 151 or rowIndex == 251 or rowIndex == 351): plotWeights(model.coef_ , 'K Ridge Regression', rowIndex)  	#Same for this - getting called at each split of training data
            predictions = model.predict(np.array(features[test]))
            printLassoRidgePerformance(linearOutput[test], predictions, output, rowIndex, C, f1_scores, "* ridgeReg *")
            temp.append(mean_squared_error(linearOutput[test],predictions))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
        f1_mean.append(np.array(f1_scores).mean())
        f1_std_dev.append(np.array(f1_scores).std())
    drawGraphs(Ci_range, mean_error, std_error, f1_mean, f1_std_dev, "ridgeReg")

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
        if(rowIndex == 21 or rowIndex == 151 or rowIndex == 251 or rowIndex == 351): plotWeights(model.coef_ , 'Ridge Regression', rowIndex)  	#Same for this - getting called at each split of training data
        predictions = model.predict(np.array(features[rowIndex-10:rowIndex+1]))
        printLassoRidgePerformance(linearOutput, predictions, output, rowIndex, C, f1_scores, "* ridgeReg *")
    print("SCORES: ", f1_scores)
    plotF1Score(Ci_range, f1_scores, 'C Values', 'F1 Score', "* ridgeReg *")
    
def plotWeights(modelWeights, modelName, iterationNumber):
    featureNames = ['Attendance', 'HomeWinStreak', 'HomeTotalPoints','HomeTotalGoalsScored', 'HomeTotalGoalsConceded', 'HomePreviousSeasonPoints', 'HomeTeamValue', 'AwayWinStreak', 'AwayTotalPoints', 'AwayTotalGoalsScored', 'AwayTotalGoalsConceded','AwayPreviousSeasonPoints','AwayTeamValue']
    outcomes = ['Away Win', 'Draw', 'Home Win']
    index = 0
    print(modelWeights)
    print(len(modelWeights))
    print(modelName)
    print(iterationNumber)
   
    if(len(modelWeights) != 13):
        for outcome in modelWeights:
            plt.figure(dpi=450)
            plt.xlabel('Feature Name')
            plt.ylabel('Feature Weight')
            plt.xticks(rotation=45)
            plt.rc('font', size=5)
            plt.rcParams['figure.constrained_layout.use'] = True
            plt.title('Weights for ' + modelName + ' model of outcome: ' + outcomes[index] + ' @ iteration no: ' + str(iterationNumber))           
            plt.plot(featureNames, outcome, color='blue', marker='x', linestyle='dotted', linewidth=1, markersize=3)
            plt.show()
            index += 1
    else:
        plt.figure(dpi=450)
        plt.xlabel('Feature Name')
        plt.ylabel('Feature Weight')
        plt.xticks(rotation=45)
        plt.rc('font', size=5)
        plt.rcParams['figure.constrained_layout.use'] = True
        plt.title('Weights for ' + modelName + ' model of outcome win draw or loss @ iteration no: ' + str(iterationNumber))
        plt.rcParams['figure.constrained_layout.use'] = True
        plt.plot(featureNames, modelWeights, color='blue', marker='x', linestyle='dotted', linewidth=1, markersize=3)
        plt.show()


def drawGraphs(Ci_range, mean_error, std_error, f1_mean, f1_std_dev, model_name):
    # print("Mean Error = ", mean_error)
    # print("Standard Deviation Error = ", std_error)
    # print("F1 Score Mean = ", f1_mean)
    # print("F1 Score Standard Deviation = ", f1_std_dev)
    graphErrorBar(Ci_range, mean_error, std_error, model_name+" 5-fold cross-validation, Mean + Standard Deviation Error Vs C")
    graphErrorBar(Ci_range, f1_mean, f1_std_dev, model_name+" 5-fold cross-validation, F1 Mean + Standard Deviation Vs C")

def plotF1Score(Clist, f1Score, xAxisTitle, yAxisTitle, title):
    plt.figure()
    plt.rc('font', size=18)
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.plot(Clist, f1Score, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
    plt.xlabel(xAxisTitle); 
    plt.ylabel(yAxisTitle); 
    plt.title(title)
    plt.show()

def graphErrorBar(Ci_range, mean_error, std_error, title):
    plt.title(title)
    plt.plot(Ci_range, mean_error)
    plt.errorbar(Ci_range, mean_error, yerr=std_error, fmt ='ro', label="Standard Deviation")
    plt.xlabel("Ci") 
    # plt.xlim((0,50))
    plt.ylabel("Mean square error")
    plt.legend()
    plt.show()

def modelTraining(features, output, rowIndex):    
    KLogisticReg(features, output, rowIndex)
    # LogisticReg(features, output, rowIndex)
    # Kknn(features, output, rowIndex)
    # knn(features, output, rowIndex)
    # randomClassifier(features, output, rowIndex)
    # KLassoReg(features, output, rowIndex)
    # lassoReg(features, output, rowIndex)
    # KRidgeReg(features, output, rowIndex)
    # ridgeReg(features, output, rowIndex)

rowIndex = 11
gameweek = 2
while gameweek < 34: #380 matches played by 20 teams 
    rowIndex+=10
    print("Gameweek: ", gameweek) 
    gameweekFeatures = np.column_stack((attendance[0:rowIndex],HomeWinStreak[0:rowIndex],HomeTotalPoints[0:rowIndex],HomeTotalGoalsScored[0:rowIndex],HomeTotalGoalsConceded[0:rowIndex],HomePreviousSeasonPoints[0:rowIndex],HomeTeamValue[0:rowIndex],AwayWinStreak[0:rowIndex],AwayTotalPoints[0:rowIndex],AwayTotalGoalsScored[0:rowIndex],AwayTotalGoalsConceded[0:rowIndex],AwayPreviousSeasonPoints[0:rowIndex],AwayTeamValue[0:rowIndex]))
    modelTraining(gameweekFeatures, matchResult[0:rowIndex], rowIndex)
    gameweek+=1