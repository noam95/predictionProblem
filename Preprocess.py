import sqlite3

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import statistics as state

def loadData(path):
    df = pd.read_csv(path)
    return df

def hundleMissingValues():
    # for col in df:
    #     df[col].fillna(state.mode(df[col]))
    pass


def normelize_data(df):
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df


def getPlayerAVGF(dat,playerApiId ,colName, tablename):
    query1= "SELECT AVG("+colName+") From " + tablename +" WHERE " + tablename +".player_api_id=" + str(playerApiId)

    # query0 = "SELECT TOP 1 * FROM (" +query1 + ") WHERE p.date< "
    query = dat.execute(query1)
    cols = [column[0] for column in query.description]
    # playerMeasurs = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
    data = query.fetchall()
    return data[0][0]


def getTeamAVGF(dat, colName, matchRow, tableName, teamType):
    sumTeamF = 0
    for i in range(11):
        x = teamType+ '_player_' + str(i+1)
        player_id = matchRow[1][x]
        if str(player_id) == 'nan':
            return 0
        sumTeamF += getPlayerAVGF(dat, player_id, colName, tableName)
    return sumTeamF/11

def getAverageFcol(matchData,dat, F, tableName):
    matchData = matchData[250:1000]
    # data = query.fetchall()
    listHomeAvg =[]
    listAwayAvg=[]
    for row in matchData.iterrows():
        home = getTeamAVGF(dat, F, row, tableName, 'home')
        listHomeAvg.append(home)
        away = getTeamAVGF(dat, F, row, tableName, 'away')
        listAwayAvg.append(away)
    matchData['avghome'] = listHomeAvg
    matchData['avgaway'] = listAwayAvg
    return matchData

# def htmlToDataFrame():


dat = sqlite3.connect("database.sqlite")
# addAverageF(dat,'39890','28/02/2009','overall_rating', 'player_Attributes')
# getAverageFcol(dat, 'overall_rating', 'player_Attributes')


df = pd.read_csv('match.csv')
newMatch = df[
    ['id', 'date', 'match_api_id', 'home_team_api_id', 'away_team_api_id', 'home_team_goal','away_team_goal', 'home_player_1', 'home_player_2',
'home_player_3','home_player_4','home_player_5','home_player_6','home_player_7','home_player_8','home_player_9', 'home_player_10', 'home_player_11', 'away_player_1',
     'away_player_2', 'away_player_3', 'away_player_4', 'away_player_5', 'away_player_6', 'away_player_7', 'away_player_8', 'away_player_9', 'away_player_10', 'away_player_11',
     ]].copy()
newMatch['result'] = newMatch['home_team_goal'] - newMatch['away_team_goal']
newMatch2 = df[['home_team_api_id', 'away_team_api_id']].copy()
getAverageFcol(newMatch, dat, 'potential', 'player_Attributes')
getAverageFcol(newMatch, dat, 'overall_rating', 'player_Attributes')
newMatch.to_excel(r'C:\Users\שי\Documents\shay\סמסטר ו\סדנת הכנה לפרויקט\חלק שלישי\newMatch.xlsx', index=False)
newMatch


norelized_df = normelize_data(newMatch)
smaller_df = norelized_df[0:1000]
data_np = smaller_df.to_numpy()
print(data_np.data)
X,y = data_np.reshape(data_np.shape[0], data_np.shape[1]), np.arange(data_np.shape[0])
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.3,random_state=42)
clf = svm.SVC(kernel='rbf')
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)
print(prediction)
print(accuracy_score(y_test, prediction))
# print("Model accurancy- " + str(91.64765)+ "%")

