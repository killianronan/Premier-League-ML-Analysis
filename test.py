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
# gameweek2 = np.column_stack((attendance[0:1],HomeWinStreak[0:1],HomeTotalPoints[0:1],HomeTotalGoalsScored[0:1],HomeTotalGoalsConceded[0:1],HomePreviousSeasonPoints[0:1],HomeTeamValue[0:1],AwayWinStreak[0:1],AwayTotalPoints[0:1],AwayTotalGoalsScored[0:1],AwayTotalGoalsConceded[0:1],AwayPreviousSeasonPoints[0:1],AwayTeamValue[0:1])) 
# gameweek3 = np.column_stack((attendance[0:2],HomeWinStreak[0:2],HomeTotalPoints[0:2],HomeTotalGoalsScored[0:2],HomeTotalGoalsConceded[0:2],HomePreviousSeasonPoints[0:2],HomeTeamValue[0:2],AwayWinStreak[0:2],AwayTotalPoints[0:2],AwayTotalGoalsScored[0:2],AwayTotalGoalsConceded[0:2],AwayPreviousSeasonPoints[0:2],AwayTeamValue[0:2])) 
# gameweek4 = np.column_stack((attendance[0:3],HomeWinStreak[0:3],HomeTotalPoints[0:3],HomeTotalGoalsScored[0:3],HomeTotalGoalsConceded[0:3],HomePreviousSeasonPoints[0:3],HomeTeamValue[0:3],AwayWinStreak[0:3],AwayTotalPoints[0:3],AwayTotalGoalsScored[0:3],AwayTotalGoalsConceded[0:3],AwayPreviousSeasonPoints[0:3],AwayTeamValue[0:3])) 
# gameweek5 = np.column_stack((attendance[0:4],HomeWinStreak[0:4],HomeTotalPoints[0:4],HomeTotalGoalsScored[0:4],HomeTotalGoalsConceded[0:4],HomePreviousSeasonPoints[0:4],HomeTeamValue[0:4],AwayWinStreak[0:4],AwayTotalPoints[0:4],AwayTotalGoalsScored[0:4],AwayTotalGoalsConceded[0:4],AwayPreviousSeasonPoints[0:4],AwayTeamValue[0:4])) 
# gameweek6 = np.column_stack((attendance[0:5],HomeWinStreak[0:5],HomeTotalPoints[0:5],HomeTotalGoalsScored[0:5],HomeTotalGoalsConceded[0:5],HomePreviousSeasonPoints[0:5],HomeTeamValue[0:5],AwayWinStreak[0:5],AwayTotalPoints[0:5],AwayTotalGoalsScored[0:5],AwayTotalGoalsConceded[0:5],AwayPreviousSeasonPoints[0:5],AwayTeamValue[0:5])) 
# gameweek7 = np.column_stack((attendance[0:6],HomeWinStreak[0:6],HomeTotalPoints[0:6],HomeTotalGoalsScored[0:6],HomeTotalGoalsConceded[0:6],HomePreviousSeasonPoints[0:6],HomeTeamValue[0:6],AwayWinStreak[0:6],AwayTotalPoints[0:6],AwayTotalGoalsScored[0:6],AwayTotalGoalsConceded[0:6],AwayPreviousSeasonPoints[0:6],AwayTeamValue[0:6])) 
# gameweek8 = np.column_stack((attendance[0:7],HomeWinStreak[0:7],HomeTotalPoints[0:7],HomeTotalGoalsScored[0:7],HomeTotalGoalsConceded[0:7],HomePreviousSeasonPoints[0:7],HomeTeamValue[0:7],AwayWinStreak[0:7],AwayTotalPoints[0:7],AwayTotalGoalsScored[0:7],AwayTotalGoalsConceded[0:7],AwayPreviousSeasonPoints[0:7],AwayTeamValue[0:7])) 
# gameweek9 = np.column_stack((attendance[0:8],HomeWinStreak[0:8],HomeTotalPoints[0:8],HomeTotalGoalsScored[0:8],HomeTotalGoalsConceded[0:8],HomePreviousSeasonPoints[0:8],HomeTeamValue[0:8],AwayWinStreak[0:8],AwayTotalPoints[0:8],AwayTotalGoalsScored[0:8],AwayTotalGoalsConceded[0:8],AwayPreviousSeasonPoints[0:8],AwayTeamValue[0:8])) 
# gameweek9 = np.column_stack((attendance[0:9],HomeWinStreak[0:9],HomeTotalPoints[0:9],HomeTotalGoalsScored[0:9],HomeTotalGoalsConceded[0:9],HomePreviousSeasonPoints[0:9],HomeTeamValue[0:9],AwayWinStreak[0:9],AwayTotalPoints[0:],AwayTotalGoalsScored[0:9],AwayTotalGoalsConceded[0:9],AwayPreviousSeasonPoints[0:9],AwayTeamValue[0:9])) 
# gameweek10 = np.column_stack((attendance[0:10],HomeWinStreak[0:10],HomeTotalPoints[0:10],HomeTotalGoalsScored[0:10],HomeTotalGoalsConceded[0:10],HomePreviousSeasonPoints[0:10],HomeTeamValue[0:10],AwayWinStreak[0:10],AwayTotalPoints[0:10],AwayTotalGoalsScored[0:10],AwayTotalGoalsConceded[0:10],AwayPreviousSeasonPoints[0:10],AwayTeamValue[0:10])) 
# gameweek11 = np.column_stack((attendance[0:11],HomeWinStreak[0:11],HomeTotalPoints[0:11],HomeTotalGoalsScored[0:11],HomeTotalGoalsConceded[0:11],HomePreviousSeasonPoints[0:11],HomeTeamValue[0:11],AwayWinStreak[0:11],AwayTotalPoints[0:11],AwayTotalGoalsScored[0:11],AwayTotalGoalsConceded[0:11],AwayPreviousSeasonPoints[0:11],AwayTeamValue[0:11])) 
# gameweek12 = np.column_stack((attendance[0:12],HomeWinStreak[0:12],HomeTotalPoints[0:12],HomeTotalGoalsScored[0:12],HomeTotalGoalsConceded[0:12],HomePreviousSeasonPoints[0:12],HomeTeamValue[0:12],AwayWinStreak[0:12],AwayTotalPoints[0:12],AwayTotalGoalsScored[0:12],AwayTotalGoalsConceded[0:12],AwayPreviousSeasonPoints[0:12],AwayTeamValue[0:12])) 
# gameweek13 = np.column_stack((attendance[0:13],HomeWinStreak[0:13],HomeTotalPoints[0:13],HomeTotalGoalsScored[0:13],HomeTotalGoalsConceded[0:13],HomePreviousSeasonPoints[0:13],HomeTeamValue[0:13],AwayWinStreak[0:13],AwayTotalPoints[0:13],AwayTotalGoalsScored[0:13],AwayTotalGoalsConceded[0:13],AwayPreviousSeasonPoints[0:13],AwayTeamValue[0:13])) 
# gameweek14 = np.column_stack((attendance[0:14],HomeWinStreak[0:14],HomeTotalPoints[0:14],HomeTotalGoalsScored[0:14],HomeTotalGoalsConceded[0:14],HomePreviousSeasonPoints[0:14],HomeTeamValue[0:14],AwayWinStreak[0:14],AwayTotalPoints[0:14],AwayTotalGoalsScored[0:14],AwayTotalGoalsConceded[0:14],AwayPreviousSeasonPoints[0:14],AwayTeamValue[0:14])) 
# gameweek15 = np.column_stack((attendance[0:15],HomeWinStreak[0:15],HomeTotalPoints[0:15],HomeTotalGoalsScored[0:15],HomeTotalGoalsConceded[0:15],HomePreviousSeasonPoints[0:15],HomeTeamValue[0:15],AwayWinStreak[0:15],AwayTotalPoints[0:15],AwayTotalGoalsScored[0:15],AwayTotalGoalsConceded[0:15],AwayPreviousSeasonPoints[0:15],AwayTeamValue[0:15])) 
# gameweek16 = np.column_stack((attendance[0:16],HomeWinStreak[0:16],HomeTotalPoints[0:16],HomeTotalGoalsScored[0:16],HomeTotalGoalsConceded[0:16],HomePreviousSeasonPoints[0:16],HomeTeamValue[0:16],AwayWinStreak[0:16],AwayTotalPoints[0:16],AwayTotalGoalsScored[0:16],AwayTotalGoalsConceded[0:16],AwayPreviousSeasonPoints[0:16],AwayTeamValue[0:16])) 
# gameweek17 = np.column_stack((attendance[0:17],HomeWinStreak[0:17],HomeTotalPoints[0:17],HomeTotalGoalsScored[0:17],HomeTotalGoalsConceded[0:17],HomePreviousSeasonPoints[0:17],HomeTeamValue[0:17],AwayWinStreak[0:17],AwayTotalPoints[0:17],AwayTotalGoalsScored[0:17],AwayTotalGoalsConceded[0:17],AwayPreviousSeasonPoints[0:17],AwayTeamValue[0:17])) 
# gameweek18 = np.column_stack((attendance[0:18],HomeWinStreak[0:18],HomeTotalPoints[0:18],HomeTotalGoalsScored[0:18],HomeTotalGoalsConceded[0:18],HomePreviousSeasonPoints[0:18],HomeTeamValue[0:18],AwayWinStreak[0:18],AwayTotalPoints[0:18],AwayTotalGoalsScored[0:18],AwayTotalGoalsConceded[0:18],AwayPreviousSeasonPoints[0:18],AwayTeamValue[0:18])) 
# gameweek19 = np.column_stack((attendance[0:19],HomeWinStreak[0:19],HomeTotalPoints[0:19],HomeTotalGoalsScored[0:19],HomeTotalGoalsConceded[0:19],HomePreviousSeasonPoints[0:19],HomeTeamValue[0:19],AwayWinStreak[0:19],AwayTotalPoints[0:19],AwayTotalGoalsScored[0:19],AwayTotalGoalsConceded[0:19],AwayPreviousSeasonPoints[0:19],AwayTeamValue[0:19])) 
# gameweek20 = np.column_stack((attendance[0:20],HomeWinStreak[0:20],HomeTotalPoints[0:20],HomeTotalGoalsScored[0:20],HomeTotalGoalsConceded[0:20],HomePreviousSeasonPoints[0:20],HomeTeamValue[0:20],AwayWinStreak[0:20],AwayTotalPoints[0:20],AwayTotalGoalsScored[0:20],AwayTotalGoalsConceded[0:20],AwayPreviousSeasonPoints[0:20],AwayTeamValue[0:20])) 
# gameweek21 = np.column_stack((attendance[0:21],HomeWinStreak[0:21],HomeTotalPoints[0:21],HomeTotalGoalsScored[0:21],HomeTotalGoalsConceded[0:21],HomePreviousSeasonPoints[0:21],HomeTeamValue[0:21],AwayWinStreak[0:21],AwayTotalPoints[0:21],AwayTotalGoalsScored[0:21],AwayTotalGoalsConceded[0:21],AwayPreviousSeasonPoints[0:21],AwayTeamValue[0:21])) 
# gameweek22 = np.column_stack((attendance[0:22],HomeWinStreak[0:22],HomeTotalPoints[0:22],HomeTotalGoalsScored[0:22],HomeTotalGoalsConceded[0:22],HomePreviousSeasonPoints[0:22],HomeTeamValue[0:22],AwayWinStreak[0:22],AwayTotalPoints[0:22],AwayTotalGoalsScored[0:22],AwayTotalGoalsConceded[0:22],AwayPreviousSeasonPoints[0:22],AwayTeamValue[0:22])) 
# gameweek23 = np.column_stack((attendance[0:23],HomeWinStreak[0:23],HomeTotalPoints[0:23],HomeTotalGoalsScored[0:23],HomeTotalGoalsConceded[0:23],HomePreviousSeasonPoints[0:23],HomeTeamValue[0:23],AwayWinStreak[0:23],AwayTotalPoints[0:23],AwayTotalGoalsScored[0:23],AwayTotalGoalsConceded[0:23],AwayPreviousSeasonPoints[0:23],AwayTeamValue[0:23])) 
# gameweek24 = np.column_stack((attendance[0:24],HomeWinStreak[0:24],HomeTotalPoints[0:24],HomeTotalGoalsScored[0:24],HomeTotalGoalsConceded[0:24],HomePreviousSeasonPoints[0:24],HomeTeamValue[0:24],AwayWinStreak[0:24],AwayTotalPoints[0:24],AwayTotalGoalsScored[0:24],AwayTotalGoalsConceded[0:24],AwayPreviousSeasonPoints[0:24],AwayTeamValue[0:24])) 
# gameweek25 = np.column_stack((attendance[0:25],HomeWinStreak[0:25],HomeTotalPoints[0:25],HomeTotalGoalsScored[0:25],HomeTotalGoalsConceded[0:25],HomePreviousSeasonPoints[0:25],HomeTeamValue[0:25],AwayWinStreak[0:25],AwayTotalPoints[0:25],AwayTotalGoalsScored[0:25],AwayTotalGoalsConceded[0:25],AwayPreviousSeasonPoints[0:25],AwayTeamValue[0:25])) 
# gameweek26 = np.column_stack((attendance[0:26],HomeWinStreak[0:26],HomeTotalPoints[0:26],HomeTotalGoalsScored[0:26],HomeTotalGoalsConceded[0:26],HomePreviousSeasonPoints[0:26],HomeTeamValue[0:26],AwayWinStreak[0:26],AwayTotalPoints[0:26],AwayTotalGoalsScored[0:26],AwayTotalGoalsConceded[0:26],AwayPreviousSeasonPoints[0:26],AwayTeamValue[0:26])) 
# gameweek27 = np.column_stack((attendance[0:27],HomeWinStreak[0:27],HomeTotalPoints[0:27],HomeTotalGoalsScored[0:27],HomeTotalGoalsConceded[0:27],HomePreviousSeasonPoints[0:27],HomeTeamValue[0:27],AwayWinStreak[0:27],AwayTotalPoints[0:27],AwayTotalGoalsScored[0:27],AwayTotalGoalsConceded[0:27],AwayPreviousSeasonPoints[0:27],AwayTeamValue[0:27])) 
# gameweek28 = np.column_stack((attendance[0:28],HomeWinStreak[0:28],HomeTotalPoints[0:28],HomeTotalGoalsScored[0:28],HomeTotalGoalsConceded[0:28],HomePreviousSeasonPoints[0:28],HomeTeamValue[0:28],AwayWinStreak[0:28],AwayTotalPoints[0:28],AwayTotalGoalsScored[0:28],AwayTotalGoalsConceded[0:28],AwayPreviousSeasonPoints[0:28],AwayTeamValue[0:28])) 
# gameweek29 = np.column_stack((attendance[0:29],HomeWinStreak[0:29],HomeTotalPoints[0:29],HomeTotalGoalsScored[0:29],HomeTotalGoalsConceded[0:29],HomePreviousSeasonPoints[0:29],HomeTeamValue[0:29],AwayWinStreak[0:29],AwayTotalPoints[0:29],AwayTotalGoalsScored[0:29],AwayTotalGoalsConceded[0:29],AwayPreviousSeasonPoints[0:29],AwayTeamValue[0:29])) 
# gameweek30 = np.column_stack((attendance[0:30],HomeWinStreak[0:30],HomeTotalPoints[0:30],HomeTotalGoalsScored[0:30],HomeTotalGoalsConceded[0:30],HomePreviousSeasonPoints[0:30],HomeTeamValue[0:30],AwayWinStreak[0:30],AwayTotalPoints[0:30],AwayTotalGoalsScored[0:30],AwayTotalGoalsConceded[0:30],AwayPreviousSeasonPoints[0:30],AwayTeamValue[0:30])) 
# gameweek31 = np.column_stack((attendance[0:31],HomeWinStreak[0:31],HomeTotalPoints[0:31],HomeTotalGoalsScored[0:31],HomeTotalGoalsConceded[0:31],HomePreviousSeasonPoints[0:31],HomeTeamValue[0:31],AwayWinStreak[0:31],AwayTotalPoints[0:31],AwayTotalGoalsScored[0:31],AwayTotalGoalsConceded[0:31],AwayPreviousSeasonPoints[0:31],AwayTeamValue[0:31])) 
# gameweek32 = np.column_stack((attendance[0:32],HomeWinStreak[0:32],HomeTotalPoints[0:32],HomeTotalGoalsScored[0:32],HomeTotalGoalsConceded[0:32],HomePreviousSeasonPoints[0:32],HomeTeamValue[0:32],AwayWinStreak[0:32],AwayTotalPoints[0:32],AwayTotalGoalsScored[0:32],AwayTotalGoalsConceded[0:32],AwayPreviousSeasonPoints[0:32],AwayTeamValue[0:32])) 
# gameweek33 = np.column_stack((attendance[0:33],HomeWinStreak[0:33],HomeTotalPoints[0:33],HomeTotalGoalsScored[0:33],HomeTotalGoalsConceded[0:33],HomePreviousSeasonPoints[0:33],HomeTeamValue[0:33],AwayWinStreak[0:33],AwayTotalPoints[0:33],AwayTotalGoalsScored[0:33],AwayTotalGoalsConceded[0:33],AwayPreviousSeasonPoints[0:33],AwayTeamValue[0:33])) 
# gameweek34 = np.column_stack((attendance[0:34],HomeWinStreak[0:34],HomeTotalPoints[0:34],HomeTotalGoalsScored[0:34],HomeTotalGoalsConceded[0:34],HomePreviousSeasonPoints[0:34],HomeTeamValue[0:34],AwayWinStreak[0:34],AwayTotalPoints[0:34],AwayTotalGoalsScored[0:34],AwayTotalGoalsConceded[0:34],AwayPreviousSeasonPoints[0:34],AwayTeamValue[0:34])) 
# gameweek35 = np.column_stack((attendance[0:35],HomeWinStreak[0:35],HomeTotalPoints[0:35],HomeTotalGoalsScored[0:35],HomeTotalGoalsConceded[0:35],HomePreviousSeasonPoints[0:35],HomeTeamValue[0:35],AwayWinStreak[0:35],AwayTotalPoints[0:35],AwayTotalGoalsScored[0:35],AwayTotalGoalsConceded[0:35],AwayPreviousSeasonPoints[0:35],AwayTeamValue[0:35])) 
# gameweek36 = np.column_stack((attendance[0:36],HomeWinStreak[0:36],HomeTotalPoints[0:36],HomeTotalGoalsScored[0:36],HomeTotalGoalsConceded[0:36],HomePreviousSeasonPoints[0:36],HomeTeamValue[0:36],AwayWinStreak[0:36],AwayTotalPoints[0:36],AwayTotalGoalsScored[0:36],AwayTotalGoalsConceded[0:36],AwayPreviousSeasonPoints[0:36],AwayTeamValue[0:36])) 
# gameweek37 = np.column_stack((attendance[0:37],HomeWinStreak[0:37],HomeTotalPoints[0:37],HomeTotalGoalsScored[0:37],HomeTotalGoalsConceded[0:37],HomePreviousSeasonPoints[0:37],HomeTeamValue[0:37],AwayWinStreak[0:37],AwayTotalPoints[0:37],AwayTotalGoalsScored[0:37],AwayTotalGoalsConceded[0:37],AwayPreviousSeasonPoints[0:37],AwayTeamValue[0:37])) 
# gameweek38 = np.column_stack((attendance[0:38],HomeWinStreak[0:38],HomeTotalPoints[0:38],HomeTotalGoalsScored[0:38],HomeTotalGoalsConceded[0:38],HomePreviousSeasonPoints[0:38],HomeTeamValue[0:38],AwayWinStreak[0:38],AwayTotalPoints[0:38],AwayTotalGoalsScored[0:38],AwayTotalGoalsConceded[0:38],AwayPreviousSeasonPoints[0:38],AwayTeamValue[0:38])) 

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
    model = LogisticRegression(max_iter=1000, C=1, penalty="l2")

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
