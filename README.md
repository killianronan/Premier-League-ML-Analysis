# Premier-League-ML-Analysis
The outcomes of football games are notoriously hard to predict, with games won by the finest of margins and the deemed ‘favourite’ never being a sure thing. Given this seemingly unpredictable nature and our interest in the sport, we decided to focus our project on the prediction of match results in the English Premier League 2018/19. We knew that finding statistics on each season on the internet would not be difficult as it is one of the most popular and media-covered competitions in the world. We trained Logistic Regression, Lasso Regression, Ridge Regression, and K-nearest neighbours classifier models, hoping to analyse, and to gauge the performance of each model with predicting matches. We will use our models to predict the outcome of matches through each gameweek, using the previous gameweeks as training data. As the training data grows with each gameweek, we hope to see better accuracy with our predictions. We also wish to investigate which features carry the highest weights, once the model has been trained i.e. which metrics have the greatest impact on the outcome of these games. We will then use a random baseline classifier to compare all of these models against. We used the following features as inputs for each of our models: 

Attendance
Home win streak
Home total points
Home total goals scored
Home total goals conceded
Home previous season points
Home team value

Away win streak
Away total points
Away total goals scored
Away total goals conceded
Away previous season points
Away team value

Each season consists of 38 gameweeks, with 10 matches played each gameweek. These input features were obtained for each match in each gameweek (380 matches), along with the result of the match which was ‘H’ for a home win, ‘A’ for an away win and ‘D’ for a draw. 
Next, starting at gameweek 2, we trained our models using the features for all games up to the current gameweek. So for gameweek 2 the input features to predict are all the matches in gameweek 2, and the training data is all the matches in gameweek 1. For gameweek 5, the training data would be all of the matches from gameweek 1 to gameweek 4 and the test data is the matches from gameweek 5, etc. Our models would then output their prediction for each match in the current gameweek.
