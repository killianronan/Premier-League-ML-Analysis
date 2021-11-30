from urllib.request import urlopen
import lxml.html as lh
import requests
import pandas as pd
import csv
import numpy as np
#from tempfile import NamedTemporaryFile
#import shutil

def webReader(pageUrl, fileName):
    #URL for scraping
    url = pageUrl
    # GET request
    page = requests.get(url)
    # Create handler
    doc = lh.fromstring(page.content)
    # Parse rows
    tableRows = doc.xpath('//tr')
    # 39 rows 
    #print(len(tableRows))
    # Initialise empty result
    column = []
    index = 0
    
    for row in tableRows[0]:
        index = index + 1
        # Get column name
        columnName=row.text_content()
        
        # Print column name
        #print("Column ", index, columnName)
        # Add column to result
        column.append((columnName,[]))
    
    # Search data from row 1 (Row 0 is column headers)
    for j in range(1,len(tableRows)):
        #i is the index of our column
        columnIndex=0
    
        #Iterate through each element of the row
        for rowElement in tableRows[j].iterchildren():
            data=rowElement.text_content() 
            data = data.strip()
            data = data.replace("Utd", "United")
            data = data.replace("F.C.", "")
            column[columnIndex][1].append(data)
            # Next column
            columnIndex= columnIndex + 1
    
    # Convert result to dictionary
    dictionaryResult = {heading:column for (heading,column) in column}
    # Convert dictionary to dataframe
    dataFrame = pd.DataFrame(dictionaryResult)
    # Convert dataframe to CSV
    dataFrame.to_csv(fileName)
    
    
def dataCleaner(scoresFileName, standingsFileName, wagesFileName):
    dataframe = pd.read_csv(scoresFileName)
    standingsDataFrame = pd.read_csv(standingsFileName)
    wagesDataFrame = pd.read_csv(wagesFileName)
    cleanedHeaders = ['Weeknumber0', 'Attendance1', 'FTHomeGoals2', 'HomeTeam3', 'HomeTeamWinStreak4', 
'HomeTeamSeasonWinsSoFar5', 'HomeTeamSeasonPointsSoFar6', 
'HomeTeamSeasonGoalsScoredSoFar7', 'HomeTeamSeasonGoalsConcededSoFar8','HomeTeamPreviousSeasonPos9', 'HomeTeamValue10', 'FTAwayGoals11', 'AwayTeam12','AwayTeamWinStreak13', 'AwayTeamSeasonWinsSoFar14', 'AwayTeamSeasonPointsSoFar15', 'AwayTeamSeasonGoalsScoredSoFar16', 'AwayTeamSeasonGoalsConcededSoFar17','AwayTeamPreviousSeasonPos18', 'AwayTeamValue19', 'MatchResult']     ##Leaving index position of columns in for the moment 
    newRowArray = []
    for index, row in dataframe.iterrows():
        
        homeTeamFullTimeGoals = str(row['Score'])[0]
        awayTeamFullTimeGoals = str(row['Score'])[len(str(row['Score'])) - 1]
        if(homeTeamFullTimeGoals == "n"): homeTeamFullTimeGoals = 0
        if(awayTeamFullTimeGoals == "n"): awayTeamFullTimeGoals = 0
        if(homeTeamFullTimeGoals != 0): homeTeamFullTimeGoals = int(homeTeamFullTimeGoals)   #Convert strings back to ints
        if(awayTeamFullTimeGoals != 0): awayTeamFullTimeGoals = int(awayTeamFullTimeGoals)
        
        matchResult = ''
        if(homeTeamFullTimeGoals > awayTeamFullTimeGoals):
            matchResult = 'H'
        elif(homeTeamFullTimeGoals < awayTeamFullTimeGoals):
            matchResult = 'A'
        elif(homeTeamFullTimeGoals == awayTeamFullTimeGoals):
            matchResult = 'D'

        homeTeamWinStreak = 0
        homeTeamWinsSoFar = 0
        homeTeamPointsSoFar = 0
        homeTeamGoalsSoFar = 0
        homeTeamGoalsConcededSoFar = 0
        
        homeTeamLastYear = standingsDataFrame.loc[standingsDataFrame['Team'].str.contains(str(row['Home']))]
        awayTeamLastYear = standingsDataFrame.loc[standingsDataFrame['Team'].str.contains(str(row['Away']))]
        
        if(homeTeamLastYear.empty): homeTeamLastYear = -1
        elif (len(homeTeamLastYear['#'].values) > 0): homeTeamLastYear = homeTeamLastYear['#'].values[0]

        if(awayTeamLastYear.empty): awayTeamLastYear = -1
        elif (len(awayTeamLastYear['#'].values) > 0): awayTeamLastYear = awayTeamLastYear['#'].values[0]
        
        homeTeamValue = wagesDataFrame.loc[wagesDataFrame['Team'].str.contains(str(row['Home']))]
        awayTeamValue = wagesDataFrame.loc[wagesDataFrame['Team'].str.contains(str(row['Away']))]
        
        
        if(homeTeamValue.empty): homeTeamValue = -1
        elif (len(homeTeamValue['Est. Total Salary'].values) > 0): 
            homeTeamValue = homeTeamValue['Est. Total Salary'].values[0]
            homeTeamValue = "£" + homeTeamValue[1:]
            #homeTeamValue = homeTeamValue.replace("Â£", "£")

        if(awayTeamValue.empty): awayTeamValue = -1
        elif (len(awayTeamValue['Est. Total Salary'].values) > 0): 
            awayTeamValue = awayTeamValue['Est. Total Salary'].values[0]
            awayTeamValue = "£" + awayTeamValue[1:]


        
        if (row['Wk'] > 1):
            prevGameOffset = 1
            while(1 == 1):
                if(dataframe.iloc[index-prevGameOffset]['Home'] == row['Home']): #if home team were home last time
                    homeTeamWinsSoFar = newRowArray[index-prevGameOffset][5]
                    homeTeamPointsSoFar = newRowArray[index-prevGameOffset][6]
                    homeTeamGoalsSoFar = newRowArray[index-prevGameOffset][7] + newRowArray[index-prevGameOffset][2]
                    homeTeamGoalsConcededSoFar = newRowArray[index-prevGameOffset][8] + newRowArray[index-prevGameOffset][11]
                    if(newRowArray[index-prevGameOffset][2] > newRowArray[index-prevGameOffset][11]): # won the game
                        homeTeamWinsSoFar += 1
                        homeTeamPointsSoFar += 3
                        homeTeamWinStreak = newRowArray[index-prevGameOffset][4] + 1
                    elif(newRowArray[index-prevGameOffset][2] == newRowArray[index-prevGameOffset][11]): #drew the game
                        homeTeamPointsSoFar += 1
                    break;
                    
                if(dataframe.iloc[index-prevGameOffset]['Away'] == row['Home']):    #home team were away last time
                    homeTeamWinsSoFar = newRowArray[index-prevGameOffset][14]
                    homeTeamPointsSoFar = newRowArray[index-prevGameOffset][15]
                    homeTeamGoalsSoFar = newRowArray[index-prevGameOffset][16] + newRowArray[index-prevGameOffset][11]
                    homeTeamGoalsConcededSoFar = newRowArray[index-prevGameOffset][17] + newRowArray[index-prevGameOffset][2]
                    if(newRowArray[index-prevGameOffset][2] < newRowArray[index-prevGameOffset][11]): #won the game
                        homeTeamWinsSoFar += 1
                        homeTeamPointsSoFar += 3
                        homeTeamWinStreak = newRowArray[index-prevGameOffset][13] + 1
                    elif(newRowArray[index-prevGameOffset][2] == newRowArray[index-prevGameOffset][11]): #drew the game
                        homeTeamPointsSoFar += 1
                    break;
                    
                if(index - prevGameOffset == 0): break 
                prevGameOffset += 1;
                
        awayTeamWinStreak = 0
        awayTeamWinsSoFar = 0
        awayTeamPointsSoFar = 0
        awayTeamGoalsSoFar = 0
        awayTeamGoalsConcededSoFar = 0
        if (row['Wk'] > 1):
            prevGameOffset = 1
            while(1 == 1):
                if(dataframe.iloc[index-prevGameOffset]['Home'] == row['Away']): #if away team were home last time
                    awayTeamWinsSoFar = newRowArray[index-prevGameOffset][5]
                    awayTeamPointsSoFar = newRowArray[index-prevGameOffset][6]
                    awayTeamGoalsSoFar = newRowArray[index-prevGameOffset][7] + newRowArray[index-prevGameOffset][2]
                    awayTeamGoalsConcededSoFar = newRowArray[index-prevGameOffset][8] + newRowArray[index-prevGameOffset][11]
                    if(newRowArray[index-prevGameOffset][2] > newRowArray[index-prevGameOffset][11]): #  and won the game
                        awayTeamWinsSoFar += 1
                        awayTeamPointsSoFar += 3
                        awayTeamWinStreak = newRowArray[index-prevGameOffset][4] + 1
                    elif(newRowArray[index-prevGameOffset][2] == newRowArray[index-prevGameOffset][11]):
                        awayTeamPointsSoFar += 1
                    break;
                if(dataframe.iloc[index-prevGameOffset]['Away'] == row['Away']): #if away team were away last time
                    awayTeamWinsSoFar = newRowArray[index-prevGameOffset][14]
                    awayTeamPointsSoFar = newRowArray[index-prevGameOffset][15]
                    awayTeamGoalsSoFar = newRowArray[index-prevGameOffset][16] + newRowArray[index-prevGameOffset][11]
                    awayTeamGoalsConcededSoFar = newRowArray[index-prevGameOffset][17] + newRowArray[index-prevGameOffset][2]
                    if(newRowArray[index-prevGameOffset][2] < newRowArray[index-prevGameOffset][11]):  #  and won the game
                        awayTeamWinsSoFar += 1
                        awayTeamPointsSoFar += 3
                        awayTeamWinStreak = newRowArray[index-prevGameOffset][13] + 1
                    elif(newRowArray[index-prevGameOffset][2] == newRowArray[index-prevGameOffset][11]):
                        awayTeamPointsSoFar += 1
                        break;
                if(index - prevGameOffset == 0): break 
                prevGameOffset += 1;

        
        newRowObj = [ row['Wk'],  row['Attendance'], homeTeamFullTimeGoals, row['Home'], homeTeamWinStreak, homeTeamWinsSoFar, homeTeamPointsSoFar, homeTeamGoalsSoFar, homeTeamGoalsConcededSoFar, homeTeamLastYear, homeTeamValue, awayTeamFullTimeGoals, row['Away'], awayTeamWinStreak, awayTeamWinsSoFar, awayTeamPointsSoFar, awayTeamGoalsSoFar, awayTeamGoalsConcededSoFar, awayTeamLastYear, awayTeamValue, matchResult]        
        newRowArray.append(newRowObj)
        
    cleanedData = pd.DataFrame(newRowArray, columns = cleanedHeaders)
    cleanedData.to_csv("cleaned" + scoresFileName);
        

webReader("https://fbref.com/en/comps/9/1889/schedule/2018-2019-Premier-League-Scores-and-Fixtures", "201819scores.csv");
webReader("https://fbref.com/en/comps/9/10728/schedule/2020-2021-Premier-League-Scores-and-Fixtures", "202021scores.csv");
webReader("https://www.skysports.com/premier-league-table/2017", "201718finalstandings.csv")
webReader("https://www.skysports.com/premier-league-table/2019", "201920finalstandings.csv")
webReader("https://www.spotrac.com/epl/payroll/2018/", "201819wages.csv")
webReader("https://www.spotrac.com/epl/payroll/2020/", "202021wages.csv")

dataCleaner("201819scores.csv", "201718finalstandings.csv", "201819wages.csv");
dataCleaner("202021scores.csv", "201920finalstandings.csv", "202021wages.csv");