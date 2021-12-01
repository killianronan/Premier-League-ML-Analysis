from re import match
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
print("HERE",matchResult)

features1819 = np.column_stack((attendance,HomeWinStreak,HomeTotalPoints,HomeTotalGoalsScored,HomeTotalGoalsConceded,HomePreviousSeasonPoints,HomeTeamValue,AwayWinStreak,AwayTotalPoints,AwayTotalGoalsScored,AwayTotalGoalsConceded,AwayPreviousSeasonPoints,AwayTeamValue)) 

def predictionsScatterPlot(X, yPredictions, title):
    plt.figure()
    plt.scatter(X[yPredictions=='D', 0], X[yPredictions=='H', 1],X[yPredictions=='A', 1], color="red")
    # plt.scatter(X[yPredictions < 0, 0], X[yPredictions < 0, 1], color="green") 
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2") 
    plt.title(title)
    plt.legend(["Positive Output","Negative Output"])    
    plt.show()

def modelTraining(features, output, gameweek, rowIndex):    
    model = LogisticRegression(max_iter=1000, C=10, penalty="l2")

    model.fit(features[0:rowIndex-10], output[0:rowIndex-10])
    predictions = model.predict(np.array(features[rowIndex-10:rowIndex+1]))

    # model.fit(features, output)
    # predictions2 = model.predict(np.array(features))

    print("Gameweek: ", gameweek) 
    print("Predicted result: ",predictions)
    print("Actual result:    ",np.array(output[rowIndex-10:rowIndex+1]))
    print("\n")

rowIndex = 11
gameweek = 2
while gameweek < 39: #380 matches played by 20 teams 
    rowIndex+=10
    gameweekFeatures = np.column_stack((attendance[0:rowIndex],HomeWinStreak[0:rowIndex],HomeTotalPoints[0:rowIndex],HomeTotalGoalsScored[0:rowIndex],HomeTotalGoalsConceded[0:rowIndex],HomePreviousSeasonPoints[0:rowIndex],HomeTeamValue[0:rowIndex],AwayWinStreak[0:rowIndex],AwayTotalPoints[0:rowIndex],AwayTotalGoalsScored[0:rowIndex],AwayTotalGoalsConceded[0:rowIndex],AwayPreviousSeasonPoints[0:rowIndex],AwayTeamValue[0:rowIndex]))
    modelTraining(gameweekFeatures, matchResult[0:rowIndex], gameweek, rowIndex)
    gameweek+=1
