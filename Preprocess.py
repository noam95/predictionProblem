import sqlite3
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import json

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


def binning(col, cut_points, labels=None):
    minval = col.min()
    maxval = col.max()
    break_point = [minval] + cut_points + [maxval]
    if not labels:
        labels = range(len(cut_points)+1)
    colBin = pd.cut(col, bins=break_point, labels=labels, include_lowest=True)
    return colBin

def getPlayerAVGF(dat,playerApiId ,colName, tablename):
    query1= "SELECT AVG("+colName+") From " + tablename +" WHERE " + tablename +".player_api_id=" + str(playerApiId)

    # query0 = "SELECT TOP 1 * FROM (" +query1 + ") WHERE p.date< "
    # query1 = "SELECT * From Match"
    query = dat.execute(query1)
    cols = [column[0] for column in query.description]
    # playerMeasurs = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
    data = query.fetchall()
    return data[0][0]

def getTeamAVGF_player(dat, colName, matchRow, tableName, teamType):
    sumTeamF = 0
    for i in range(11):
        x = teamType+ '_player_' + str(i+1)
        player_id = matchRow[1][x]
        if str(player_id) == 'nan':
            return 0
        sumTeamF += getPlayerAVGF(dat, player_id, colName, tableName)
    return sumTeamF/11

def getAverageFcol_players(matchData,dat, F, tableName):
    # data = query.fetchall()
    listHomeAvg =[]
    listAwayAvg=[]
    for row in matchData.iterrows():
        home = getTeamAVGF_player(dat, F, row, tableName, 'home')
        listHomeAvg.append(home)
        away = getTeamAVGF_player(dat, F, row, tableName, 'away')
        listAwayAvg.append(away)
    home_name = F +'home'
    away_name = F + 'away'
    matchData[home_name] = listHomeAvg
    matchData[away_name] = listAwayAvg
    return matchData

def getTeamF(dat, F, team_api_id):
    query1= "SELECT AVG("+F+") From Team_Attributes WHERE Team_Attributes.team_api_id=" + str(team_api_id)
    query = dat.execute(query1)
    data = query.fetchall()
    return data[0][0]

def getAverageFcol_team(df,dat, F):
    listHome =[]
    listAway=[]
    for row in df.iterrows():
        home = getTeamF(dat, F, row[1]['home_team_api_id'])
        listHome.append(home)
        away = getTeamF(dat, F, row[1]['away_team_api_id'])
        listAway.append(away)
    home_name = F +'home'
    away_name = F + 'away'
    df[home_name] = listHome
    df[away_name] = listAway
    return df

def fromXML(table):
    for row in table.iterrows():
        x = row[1]['shoton']
        root = ET.fromstring(x)
        y = root.tag
        z = root.attrib
        # ET.etree.ElementTree.fromstring
        # for child in root:
        #     print(child.tag, child.attrib)
        if str(table[1]['shoton']) == 'nan':
            return 0
        table[1]['shoton'].readxml

def resultCol(df):
    df['result'] = df['home_team_goal'] - df['away_team_goal']
    bins= [-0.5, 0.5]
    group_names = ['0', 'teco', '1']
    df["class_res"] = binning(df['result'], bins, group_names)
    df = df[df.class_res != 'teco']
    return df

def orderData():
    dat = sqlite3.connect("database.sqlite")
    tables = "id, date, match_api_id, home_team_api_id, away_team_api_id, home_team_goal, away_team_goal, home_player_1, home_player_2, home_player_3, home_player_4, home_player_5, home_player_6, home_player_7, home_player_8, home_player_9, home_player_10, home_player_11, away_player_1, away_player_2, away_player_3, away_player_4, away_player_5, away_player_6, away_player_7, away_player_8, away_player_9, away_player_10, away_player_11"
    query = dat.execute("SELECT " + tables + " From Match")
    cols = [column[0] for column in query.description]
    df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
    # df = pd.read_csv('match.csv')
    # df = df[
    #     ['id', 'date', 'match_api_id', 'home_team_api_id', 'away_team_api_id', 'home_team_goal', 'away_team_goal',
    #      'home_player_1', 'home_player_2',
    #      'home_player_3', 'home_player_4', 'home_player_5', 'home_player_6', 'home_player_7', 'home_player_8',
    #      'home_player_9', 'home_player_10', 'home_player_11', 'away_player_1',
    #      'away_player_2', 'away_player_3', 'away_player_4', 'away_player_5', 'away_player_6', 'away_player_7',
    #      'away_player_8', 'away_player_9', 'away_player_10', 'away_player_11',
    #      'shoton'
    #      ]].copy()
    # newMatch=newMatch[1750:2000]
    df = resultCol(df)
    df = df[350:400]
    # fromXML(newMatch)
    df = getAverageFcol_team(df, dat, 'defencePressure')
    df = getAverageFcol_team(df, dat, 'buildUpPlaySpeed')
    df = getAverageFcol_team(df, dat, 'buildUpPlayPassing')
    df = getAverageFcol_team(df, dat, 'chanceCreationPassing')
    df = getAverageFcol_team(df, dat, 'chanceCreationCrossing')
    df = getAverageFcol_team(df, dat, 'chanceCreationShooting')
    df = getAverageFcol_team(df, dat, 'defencePressure')
    df = getAverageFcol_team(df, dat, 'defenceAggression')
    df = getAverageFcol_team(df, dat, 'defenceTeamWidth')
    df = getAverageFcol_players(df, dat, 'crossing', 'player_Attributes')
    df = getAverageFcol_players(df, dat, 'finishing', 'player_Attributes')
    df = getAverageFcol_players(df, dat, 'heading_accuracy', 'player_Attributes')
    df = getAverageFcol_players(df, dat, 'volleys', 'player_Attributes')
    df = getAverageFcol_players(df, dat, 'dribbling', 'player_Attributes')
    df = getAverageFcol_players(df, dat, 'curve', 'player_Attributes')
    df = getAverageFcol_players(df, dat, 'long_passing', 'player_Attributes')
    df = getAverageFcol_players(df, dat, 'aggression', 'player_Attributes')
    df = getAverageFcol_players(df, dat, 'short_passing', 'player_Attributes')
    df = getAverageFcol_players(df, dat, 'potential', 'player_Attributes')
    df = getAverageFcol_players(df, dat, 'overall_rating', 'player_Attributes')
    df = getAverageFcol_players(df, dat, 'long_shots', 'player_Attributes')
    df = getAverageFcol_players(df, dat, 'ball_control', 'player_Attributes')
    df.to_excel(r'newMatch.xlsx', index=False)
    # df = normelize_data(df)

    return df



orderData()








# norelized_df = normelize_data(newMatch)
# smaller_df = norelized_df[0:1000]
# data_np = smaller_df.to_numpy()
# print(data_np.data)
# X,y = data_np.reshape(data_np.shape[0], data_np.shape[1]), np.arange(data_np.shape[0])
# x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.3,random_state=42)
# clf = svm.SVC(kernel='rbf')
# clf.fit(x_train, y_train)
# prediction = clf.predict(x_test)
# print(prediction)
# print(accuracy_score(y_test, prediction))
# print("Model accurancy- " + str(91.64765)+ "%")

####

# newMatch['result'] = newMatch['home_team_goal'] - newMatch['away_team_goal']
# # newMatch2 = df[['home_team_api_id', 'away_team_api_id']].copy()
# bins= [-0.5, 0.5]
# group_names = ['0', 'teco', '1']
# newMatch["class_res"] = binning(newMatch['result'], bins, group_names)
# match_class = newMatch[newMatch.class_res != 'teco']
# # print(match_class)


