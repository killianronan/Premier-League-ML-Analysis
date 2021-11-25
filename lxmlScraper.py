from urllib.request import urlopen
import lxml.html as lh
import requests
import pandas as pd

#URL for scraping
url = "https://fbref.com/en/squads/4ba7cbea/2018-2019/matchlogs/s1889/schedule/Bournemouth-Scores-and-Fixtures-Premier-League"
# GET request
page = requests.get(url)
# Create handler
doc = lh.fromstring(page.content)
# Parse rows
tableRows = doc.xpath('//tr')
# 39 rows 
print(len(tableRows))
# Initialise empty result
column = []
index = 0

for row in tableRows[0]:
    index = index + 1
    # Get column name
    columnName=row.text_content()
    # Print column name
    print("Column ", index, columnName)
    # Add column to result
    column.append((columnName,[]))

# Search data from row 1 (Row 0 is column headers)
for j in range(1,len(tableRows)):
    #i is the index of our column
    columnIndex=0

    #Iterate through each element of the row
    for rowElement in tableRows[j].iterchildren():
        data=rowElement.text_content() 
        # Add data to result
        column[columnIndex][1].append(data)
        # Next column
        columnIndex= columnIndex + 1

# Convert result to dictionary
dictionaryResult = {heading:column for (heading,column) in column}
# Convert dictionary to dataframe
dataFrame = pd.DataFrame(dictionaryResult)
# Convert dataframe to CSV
dataFrame.to_csv("bournemouth.csv")