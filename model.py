import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import ConfusionMatrixDisplay

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

def calculate2x2Metrics(confusion_m):
    tp = confusion_m[0]
    tn = confusion_m[3]
    fp = confusion_m[2]
    fn = confusion_m[1]
    if((tn+tp)!=0 and (tn + fp + tp + fn)!=0):
        accuracy = (tn + tp)/(tn + fp + tp + fn)
    else: accuracy = 0
    if(tp!=0 and (tp + fp)!=0):
        precision = (tp)/(tp + fp)
    else: precision = 0
    if(tp!=0 and (tp + fn)!=0):
        recall = (tp)/(tp + fn)
    else: recall = 0
    if(precision != 0 and recall != 0):
        f1Score = 2 * ((precision * recall)/(precision + recall))
    else: f1Score = 0

    return f1Score, accuracy

def calculate3x3Metrics(confusion_m): 
    tpA = confusion_m[0]
    tnA = confusion_m[4]+confusion_m[5]+confusion_m[7]+confusion_m[8]
    fpA = confusion_m[3]+confusion_m[6]
    fnA = confusion_m[1]+confusion_m[2]
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

    tpH = confusion_m[8]
    tnH = confusion_m[0]+confusion_m[1]+confusion_m[3]+confusion_m[4]
    fpH = confusion_m[2]+confusion_m[5]
    fnH = confusion_m[6]+confusion_m[7]
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

    tpD = confusion_m[4]
    tnD = confusion_m[0]+confusion_m[2]+confusion_m[6]+confusion_m[8]
    fpD = confusion_m[1]+confusion_m[7]
    fnD = confusion_m[3]+confusion_m[5]
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

    if((accuracyA + accuracyH + accuracyD) != 0):
        accuracyTotal = (accuracyA + accuracyH + accuracyD) / 3
    else: accuracyTotal = 0
    if((f1ScoreA + f1ScoreH + f1ScoreD) != 0):
        f1ScoreTotal = (f1ScoreA + f1ScoreH + f1ScoreD) / 3
    else: f1ScoreTotal = 0
    if((precisionA + precisionH + precisionD) != 0):
        precisionTotal = (precisionA + precisionH + precisionD) / 3
    else: precisionTotal = 0
    if((recallA + recallH + recallD) != 0):
        recallTotal = (recallA + recallH + recallD) / 3
    else: recallTotal = 0
    print("Precision: ", precisionTotal)
    print("Recall: ", recallTotal)
    print("F1: ", f1ScoreTotal)
    print("Accuracy: ", accuracyTotal)

    return f1ScoreTotal, accuracyTotal

def calculateLogisticKNNPerformance(predictions, output, f1_scores, accuracy_scores):
    confusion_m = confusion_matrix(np.array(output), predictions).ravel()
    if(len(confusion_m)==9):
        newF1, newAccuracy = calculate3x3Metrics(confusion_m)
        f1_scores.append(newF1)
        accuracy_scores.append(newAccuracy)
    elif(len(confusion_m)==4):
        newF1, newAccuracy = calculate2x2Metrics(confusion_m)
        f1_scores.append(newF1)
        accuracy_scores.append(newAccuracy)

def calculateLassoRidgePerformance(linearOutput, predictions, f1_scores, accuracy_scores):
    predictions[predictions<=-.33] = -1
    predictions[predictions>=.33] = 1
    predictions[abs(predictions)<1] = 0
    result = np.zeros((len(predictions), 1))
    for i in range(len(predictions)):
        result[i] = linearOutput[i]
    confusion_m = confusion_matrix(result, predictions).ravel()
    if(len(confusion_m)==9):
        newF1, newAccuracy = calculate3x3Metrics(confusion_m)
        f1_scores.append(newF1)
        accuracy_scores.append(newAccuracy)

def convertCharToNumber(actual, predictions):
    actualOutput = np.where(actual == 'A', -1.0, actual)
    actualOutput = np.where(actualOutput == 'H', 1.0, actualOutput)
    actualOutput = np.where(actualOutput == 'D', 0.0, actualOutput)
    output = np.where(predictions == 'A', -1.0, predictions)
    output = np.where(output == 'H', 1.0, output)
    output = np.where(output == 'D', 0.0, output)
    return actualOutput, output

def KLogisticReg(features, output, rowIndex): 
    Ci_range = [0.001, 0.01, 0.1, 1, 10, 100]
    mean_error=[]; std_error=[]; f1_mean = []; f1_std_dev = []; accuracy_mean = []; accuracy_std_dev = []
    for C in Ci_range:
        model = LogisticRegression(max_iter=1000, C = C, penalty="l2")
        temp = []; f1_scores = []; accuracy_scores = []
        kf = KFold(n_splits=5)
        for train, test in kf.split(features[0:rowIndex-10]):
            model.fit(features[train], output[train])
            #if(rowIndex == 21 or rowIndex == 151 or rowIndex == 251 or rowIndex == 351): plotWeights(model.coef_ , 'K-Fold Logistic Regression', rowIndex)
            predictions = model.predict(np.array(features[test]))
            calculateLogisticKNNPerformance(predictions, output[test], f1_scores, accuracy_scores)
            convertedActual, convertedPredictions = convertCharToNumber(output,predictions)
            temp.append(mean_squared_error(convertedActual[test],convertedPredictions))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
        f1_mean.append(np.array(f1_scores).mean())
        f1_std_dev.append(np.array(f1_scores).std())
        accuracy_mean.append(np.array(accuracy_scores).mean())
        accuracy_std_dev.append(np.array(accuracy_scores).std())
    graphErrorBar(Ci_range, mean_error, std_error, "Logistic - Mean Squared Error", "Mean Squared", "Ci")
    graphErrorBar(Ci_range, f1_mean, f1_std_dev, "Logistic - F1 Score", "F1 Score", "Ci")
    graphErrorBar(Ci_range, accuracy_mean, accuracy_std_dev, "Logistic - Accuracy Score", "Accuracy Score", "Ci")

def logisticReg(features, output, rowIndex): 
    Ci_range = [1]
    f1_scores = []; accuracy_scores = []
    for C in Ci_range:
        model = LogisticRegression( C = C, penalty="l2")
        model.fit(features[0:rowIndex-10], output[0:rowIndex-10])
        print("Intercept = ", model.intercept_)
        print("Coefficient = ", model.coef_)
        # if(rowIndex == 21 or rowIndex == 151 or rowIndex == 251 or rowIndex == 351): plotWeights(model.coef_ , 'Logistic Regression', rowIndex)  ## NEED TO EITHER PICK A C VALUE TO STICK WITH OR PASS C VALUE THROUGH TO THIS FUNCTION
        predictions = model.predict(np.array(features[rowIndex-10:rowIndex+1]))
        
        calculateLogisticKNNPerformance(predictions, output[rowIndex-10:rowIndex+1], f1_scores, accuracy_scores)
    # plotPerformance(Ci_range, f1_scores, 'C Values', 'F1 Score', "Logistic F1 Performance")
    plotPerformance(Ci_range, accuracy_scores, 'C Values', 'Accuracy Score', "Logistic Accuracy Performance")

def Kknn(features, output, rowIndex): 
    Ki_range = [2, 3, 4, 5, 6]
    mean_error=[]; std_error=[]; f1_mean = []; f1_std_dev = []; accuracy_mean = []; accuracy_std_dev = []
    for n_neighbours in Ki_range:
        model = KNeighborsClassifier(n_neighbours, weights='uniform')
        temp = []; f1_scores = []; accuracy_scores = []
        kf = KFold(n_splits=5)
        for train, test in kf.split(features[0:rowIndex-80]):
            model.fit(features[train], output[train])
            #if(rowIndex == 21 or rowIndex == 151 or rowIndex == 251 or rowIndex == 351): plotWeights(model.coef_ , 'KNN', rowIndex) #Need to fix this one up a bit will do tomorrow
            predictions = model.predict(np.array(features[test]))
            calculateLogisticKNNPerformance(predictions, output[test], f1_scores, accuracy_scores)
            convertedActual, convertedPredictions = convertCharToNumber(output,predictions)
            temp.append(mean_squared_error(convertedActual[test],convertedPredictions))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
        f1_mean.append(np.array(f1_scores).mean())
        f1_std_dev.append(np.array(f1_scores).std())
        accuracy_mean.append(np.array(accuracy_scores).mean())
        accuracy_std_dev.append(np.array(accuracy_scores).std())
    graphErrorBar(Ki_range, mean_error, std_error, "KNN - Mean Squared Error", "Mean Squared", "Ki")
    # graphErrorBar(Ki_range, f1_mean, f1_std_dev, "KNN - F1 Score", "F1 Score", "Ki")
    # graphErrorBar(Ki_range, accuracy_mean, accuracy_std_dev, "KNN - Accuracy Score", "Accuracy Score", "Ki")
    graphAccuracyF1Error(Ki_range, f1_mean, f1_std_dev, accuracy_mean, accuracy_std_dev, "KNN - Accuracy Score", "Accuracy Score", "Ki")


def knn(features, output, rowIndex): 
    # Ki_range = [2, 3, 4, 5, 6]
    Ki_range = [5]
    f1_scores = []; accuracy_scores = []
    for n_neighbours in Ki_range:
        model = KNeighborsClassifier(n_neighbours, weights='uniform')
        model.fit(features[0:rowIndex-80], output[0:rowIndex-80])
        predictions = model.predict(np.array(features[rowIndex-80:rowIndex+1]))
        calculateLogisticKNNPerformance(predictions, output[rowIndex-80:rowIndex+1], f1_scores, accuracy_scores)
        ConfusionMatrixDisplay.from_predictions(np.array(output[rowIndex-80:rowIndex+1]), predictions)
        plt.title(str(n_neighbours) + " Neighbours")
        plt.show()
    plotPerformance(Ki_range, f1_scores, 'K Values', 'F1 Score', "F1 Performance")
    plotPerformance(Ki_range, accuracy_scores, 'K Values', 'Accuracy Score', "Accuracy Performance")

def randomClassifier(features, output, rowIndex): 
    dummy_clf = DummyClassifier(strategy="uniform")
    dummy_clf.fit(features[0:rowIndex-80], output[0:rowIndex-80])
    f1_scores = []; accuracy_scores = []
    predictions = dummy_clf.predict(np.array(features[rowIndex-80:rowIndex+1]))
    calculateLogisticKNNPerformance(predictions, output[rowIndex-80:rowIndex+1], f1_scores, accuracy_scores)
    # print("F1 Score: ", f1_scores[0])
    # print("Accuracy Score: ", accuracy_scores[0])

def KLassoReg(features, output, rowIndex): 
    linearOutput = np.where(output == 'A', -1.0, output)
    linearOutput = np.where(linearOutput == 'H', 1.0, linearOutput)
    linearOutput = np.where(linearOutput == 'D', 0.0, linearOutput)
    Ci_range = [0.5, 1, 5, 10, 50, 100]
    mean_error=[]; std_error=[]; f1_mean = []; f1_std_dev = []; accuracy_mean = []; accuracy_std_dev = []
    for C in Ci_range:
        model = linear_model.Lasso(alpha=1/(2*C), max_iter=2000)
        temp = []; f1_scores = []; accuracy_scores = []
        kf = KFold(n_splits=5)
        for train, test in kf.split(features[0:rowIndex-10]):
            model.fit(features[train], linearOutput[train])
            # if(rowIndex == 21 or rowIndex == 151 or rowIndex == 251 or rowIndex == 351): plotWeights(model.coef_ , 'K-Lasso Regression', rowIndex)
            predictions = model.predict(np.array(features[test]))
            calculateLassoRidgePerformance(linearOutput[test], predictions, f1_scores, accuracy_scores)
            temp.append(mean_squared_error(linearOutput[test],predictions))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
        f1_mean.append(np.array(f1_scores).mean())
        f1_std_dev.append(np.array(f1_scores).std())
        accuracy_mean.append(np.array(accuracy_scores).mean())
        accuracy_std_dev.append(np.array(accuracy_scores).std())
    graphErrorBar(Ci_range, mean_error, std_error, "Lasso - Mean Squared Error", "Mean Squared", "Ci")
    # graphErrorBar(Ci_range, f1_mean, f1_std_dev, "Lasso - F1 Score", "F1 Score", "Ki")
    # graphErrorBar(Ci_range, accuracy_mean, accuracy_std_dev, "Lasso - Accuracy Score", "Accuracy Score", "Ki")
    graphAccuracyF1Error(Ci_range, f1_mean, f1_std_dev, accuracy_mean, accuracy_std_dev, "Lasso - Accuracy Score", "Accuracy Score", "Ci")


def lassoReg(features, output, rowIndex): 
    linearOutput = np.where(output == 'A', -1, output)
    linearOutput = np.where(linearOutput == 'H', 1, linearOutput)
    linearOutput = np.where(linearOutput == 'D', 0, linearOutput)
    # Ci_range = [0.5, 1, 5, 10, 50, 100]
    Ci_range = [1, 10]
    f1_scores = []; accuracy_scores = []
    for C in Ci_range:
        model = linear_model.Lasso(alpha=1/(2*C))
        model.fit(features[0:rowIndex-80], linearOutput[0:rowIndex-80])
        print("Intercept = ", model.intercept_)
        print("Coefficient = ", model.coef_)
        # if(rowIndex == 21 or rowIndex == 151 or rowIndex == 251 or rowIndex == 351): plotWeights(model.coef_ , 'Lasso Regression', rowIndex)
        predictions = model.predict(np.array(features[rowIndex-80:rowIndex+1]))
        calculateLassoRidgePerformance(linearOutput, predictions, f1_scores, accuracy_scores)
        print(linearOutput[rowIndex-80:rowIndex+1])
        print(predictions)
        testChar = np.where(linearOutput == -1, 'A', linearOutput)
        testChar = np.where(testChar == 1, 'H', testChar)
        testChar = np.where(testChar == 0, 'D', testChar)

        testChar2 = np.where(predictions == -1.0, 'A', predictions)
        testChar2 = np.where(testChar2 == '1.0', 'H', testChar2)
        testChar2 = np.where(testChar2 == '0.0', 'D', testChar2)
        ConfusionMatrixDisplay.from_predictions(testChar[rowIndex-80:rowIndex+1], testChar2)
        plt.title("C = " + str(C))
        plt.show()
    plotPerformance(Ci_range, f1_scores, 'C Values', 'F1 Score', "Lasso - F1 Performance")
    plotPerformance(Ci_range, accuracy_scores, 'C Values', 'Accuracy Score', "Lasso - Accuracy Performance")

def KRidgeReg(features, output, rowIndex): 
    linearOutput = np.where(output == 'A', -1.0, output)
    linearOutput = np.where(linearOutput == 'H', 1.0, linearOutput)
    linearOutput = np.where(linearOutput == 'D', 0.0, linearOutput)
    Ci_range = [0.5, 1, 5, 10, 50, 100]
    mean_error=[]; std_error=[]; f1_mean = []; f1_std_dev = []; accuracy_mean = []; accuracy_std_dev = []
    for C in Ci_range:
        model = linear_model.Ridge(alpha=1/(2*C))
        temp = []; f1_scores = []; accuracy_scores = []
        kf = KFold(n_splits=5)
        for train, test in kf.split(features[0:rowIndex-10]):
            model.fit(features[train], linearOutput[train])
            # if(rowIndex == 21 or rowIndex == 151 or rowIndex == 251 or rowIndex == 351): plotWeights(model.coef_ , 'K Ridge Regression', rowIndex)  	#Same for this - getting called at each split of training data
            predictions = model.predict(np.array(features[test]))
            calculateLassoRidgePerformance(linearOutput[test], predictions, f1_scores, accuracy_scores)
            temp.append(mean_squared_error(linearOutput[test],predictions))
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std())
        f1_mean.append(np.array(f1_scores).mean())
        f1_std_dev.append(np.array(f1_scores).std())
        accuracy_mean.append(np.array(accuracy_scores).mean())
        accuracy_std_dev.append(np.array(accuracy_scores).std())
    graphErrorBar(Ci_range, mean_error, std_error, "Ridge - Mean Squared Error", "Mean Squared", "Ki")
    graphErrorBar(Ci_range, f1_mean, f1_std_dev, "Ridge - F1 Score", "F1 Score", "Ki")
    graphErrorBar(Ci_range, accuracy_mean, accuracy_std_dev, "Ridge - Accuracy Score", "Accuracy Score", "Ki")

def ridgeReg(features, output, rowIndex): 
    linearOutput = np.where(output == 'A', -1.0, output)
    linearOutput = np.where(linearOutput == 'H', 1.0, linearOutput)
    linearOutput = np.where(linearOutput == 'D', 0.0, linearOutput)
    Ci_range = [0.5, 1, 5, 10, 50, 100]
    f1_scores = []; accuracy_scores = []
    for C in Ci_range:
        model = linear_model.Ridge(alpha=1/(2*C))
        model.fit(features[0:rowIndex-10], linearOutput[0:rowIndex-10])
        # if(rowIndex == 21 or rowIndex == 151 or rowIndex == 251 or rowIndex == 351): plotWeights(model.coef_ , 'Ridge Regression', rowIndex)  	#Same for this - getting called at each split of training data
        predictions = model.predict(np.array(features[rowIndex-10:rowIndex+1]))
        calculateLassoRidgePerformance(linearOutput, predictions, f1_scores, accuracy_scores)
    plotPerformance(Ci_range, f1_scores, 'C Values', 'F1 Score', "Ridge - F1 Performance")
    plotPerformance(Ci_range, accuracy_scores, 'C Values', 'Accuracy Score', "Ridge - Accuracy Performance")
    
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
    
def plotPerformance(Ci_range, scores, xAxisTitle, yAxisTitle, title):
    plt.figure()
    plt.rc('font', size=18)
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.plot(Ci_range, scores, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
    plt.xlabel(xAxisTitle); 
    plt.ylabel(yAxisTitle); 
    plt.title(title)
    plt.show()

def graphErrorBar(Ci_range, mean_error, std_error, title, label, xLabel):
    plt.title(title)
    plt.plot(Ci_range, mean_error)
    plt.errorbar(Ci_range, mean_error, yerr=std_error, fmt ='ro', label=label)
    plt.xlabel(xLabel) 
    # plt.xlim((0,50))
    plt.ylabel("Mean square error")
    plt.legend()
    plt.show()

def graphAccuracyF1Error(Ci_range, mean_error, std_error, mean_error2, std_error2, title, label, xLabel):
    plt.title(title)
    plt.plot(Ci_range, mean_error)
    plt.plot(Ci_range, mean_error2)
    plt.errorbar(Ci_range, mean_error, yerr=std_error, fmt ='ro', label="F1", ecolor='r', color='red')
    plt.errorbar(Ci_range, mean_error2, yerr=std_error2, fmt ='ro', label="Accuracy", ecolor='black', color='black')
    plt.xlabel(xLabel) 
    plt.ylabel("F1 Accuracy & Accuracy")
    plt.legend()
    plt.show()

def modelTraining(features, output, rowIndex):    
    # KLogisticReg(features, output, rowIndex)
    # logisticReg(features, output, rowIndex)
    # Kknn(features, output, rowIndex)
    knn(features, output, rowIndex)
    # randomClassifier(features, output, rowIndex)
    # KLassoReg(features, output, rowIndex)
    # lassoReg(features, output, rowIndex)
    # KRidgeReg(features, output, rowIndex)
    # ridgeReg(features, output, rowIndex)

rowIndex = 361
gameweek = 37
while gameweek < 39: #380 matches played by 20 teams 
    rowIndex+=10
    print("Gameweek: ", gameweek) 
    gameweekFeatures = np.column_stack((attendance[0:rowIndex],HomeWinStreak[0:rowIndex],HomeTotalPoints[0:rowIndex],HomeTotalGoalsScored[0:rowIndex],HomeTotalGoalsConceded[0:rowIndex],HomePreviousSeasonPoints[0:rowIndex],HomeTeamValue[0:rowIndex],AwayWinStreak[0:rowIndex],AwayTotalPoints[0:rowIndex],AwayTotalGoalsScored[0:rowIndex],AwayTotalGoalsConceded[0:rowIndex],AwayPreviousSeasonPoints[0:rowIndex],AwayTeamValue[0:rowIndex]))
    modelTraining(gameweekFeatures, matchResult[0:rowIndex], rowIndex)
    gameweek+=1