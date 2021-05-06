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

def hundleMissingValues(df,F):
    # for col in df:
    #     df[col].fillna(state.mode(df[col]))
    x = df[df[F] != 0]
    avg = x[F].mean()
    df[F].replace({0: avg}, inplace=True)
    return df

def zero_handle(df):
    # col_with_zero = ['ball_control_away', 'ball_control_home', 'long_shots_away', 'long_shots_away', 'long_shots_home',
    #                  'crossing_home','crossing_away', 'finishing_home','finishing_away', 'heading_accuracy_home', 'heading_accuracy_home', 'volleys_home', 'volleys_away', 'dribbling_home', 'dribbling_away',
    #                  'curve_home', 'curve_away', 'long_passing_home', 'long_passing_away', 'aggression_home', 'aggression_away', 'short_passing_home', 'short_passing_away', 'potential_home',
    #                  'potential_away', 'overall_rating_home','overall_rating_away', 'long_shots_home', 'long_shots_away', 'ball_control_home', 'ball_control_away']
    col_with_zero = ['ball_control', 'long_shots', 'crossing', 'finishing', 'heading_accuracy', 'volleys', 'dribbling',
                     'curve', 'long_passing', 'aggression', 'short_passing', 'potential', 'overall_rating',
                     'long_shots',
                     'ball_control']
    for col in col_with_zero:
        hundleMissingValues(df, col)
    return df

def normelize_data(df):
    # result = df.copy()
    for feature_name in df.columns:
        if feature_name not in ['home_team_api_id', 'away_team_api_id', 'class_res','season']:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            df[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return df

def binning(col, cut_points, labels=None):
    minval = col.min()
    maxval = col.max()
    break_point = [minval] + cut_points + [maxval]
    if not labels:
        labels = range(len(cut_points)+1)
    colBin = pd.cut(col, bins=break_point, labels=labels, include_lowest=True)
    return colBin

def getPlayerAVGF_old(dat,playerApiId ,colName, tablename):
    query_string = "SELECT AVG("+colName+") From " + tablename +" WHERE " + tablename +".player_api_id=" + str(playerApiId)

    # query0 = "SELECT TOP 1 * FROM (" +query1 + ") WHERE p.date< "
    # query1 = "SELECT * From Match"
    query = dat.execute(query_string)
    # cols = [column[0] for column in query.description]
    # playerMeasurs = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
    data = query.fetchall()
    if data[0][0] is None:
        return 0
    return data[0][0]

def getPlayerAVGF(df,playerApiId ,colName):
    df = df[df['player_api_id'] == playerApiId]
    mean = df[colName].mean()

    if str(mean) == 'nan':
        return 0
    return mean

def getTeamAVGF_player(palyersDF, colName, matchRow, teamType):
    sumTeamF = 0
    sum_player=0
    for i in range(11):
        x = teamType+ '_player_' + str(i+1)
        player_id = matchRow[1][x]
        if str(player_id) == 'nan':
            continue
        sum_player += 1
        # sumTeamF += getPlayerAVGF(dat, player_id, colName, tableName)
        sumTeamF += getPlayerAVGF(palyersDF, player_id, colName)
    if sumTeamF != 0:
        return sumTeamF/sum_player
    return 0

def getAverageFcol_players(matchData,dat, F):
    # data = query.fetchall()
    # listHomeAvg =[]
    # listAwayAvg=[]
    listData=[]
    query0 = "SELECT * FROM Player_Attributes "
    query = dat.execute(query0)
    cols = [column[0] for column in query.description]
    palyersDF = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)

    for row in matchData.iterrows():
        home = getTeamAVGF_player(palyersDF, F, row, 'home')
        # listHomeAvg.append(home)
        away = getTeamAVGF_player(palyersDF, F, row, 'away')
        # listAwayAvg.append(away)
        listData.append(home-away)
    # home_name = F +'_home'
    # away_name = F + '_away'
    # matchData[home_name] = listHomeAvg
    # matchData[away_name] = listAwayAvg
    matchData[F] = listData
    return matchData

def getTeamF(df, F, team_api_id):
    # query1= "SELECT AVG("+F+") From Team_Attributes WHERE Team_Attributes.team_api_id=" + str(team_api_id)
    # query = dat.execute(query1)
    # data = query.fetchall()
    # return data[0][0]

    df = df[df['team_api_id'] == team_api_id]
    mean = df[F].mean()

    if str(mean) =='nan':
        return 0
    return mean

def getAverageFcol_team(df,dat, F):
    # listHome =[]
    # listAway=[]
    listData=[]
    query0 = "SELECT * FROM Team_Attributes "
    query = dat.execute(query0)
    cols = [column[0] for column in query.description]
    teamDF = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
    for row in df.iterrows():
        home = getTeamF(teamDF, F, row[1]['home_team_api_id'])
        # listHome.append(home)
        away = getTeamF(teamDF, F, row[1]['away_team_api_id'])
        # listAway.append(away)
        listData.append(home-away)
    # home_name = F +'_home'
    # away_name = F + '_away'
    # df[home_name] = listHome
    # df[away_name] = listAway
    df[F]= listData
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

def remove_not_data_good(df):
    list_to_drop = ['home_player_1', 'home_player_2', 'home_player_3', 'home_player_4','home_player_5','home_player_6','home_player_7',
                    'home_player_8', 'home_player_9', 'home_player_10', 'home_player_11', 'away_player_1', 'away_player_2', 'away_player_3', 'away_player_4', 'away_player_5',
                    'away_player_6', 'away_player_7', 'away_player_8','away_player_9', 'away_player_10', 'away_player_11', 'result']
    df.drop(list_to_drop, inplace=True, axis=1)

def trainTest(df):
    train_table = df[df.season != '2015/2016']
    test_table = df[df.season == '2015/2016']
    return train_table, test_table

def addF(df,dat):
    df = getAverageFcol_team(df, dat, 'defencePressure')
    df = getAverageFcol_team(df, dat, 'buildUpPlaySpeed')
    df = getAverageFcol_team(df, dat, 'buildUpPlayPassing')
    df = getAverageFcol_team(df, dat, 'chanceCreationPassing')
    df = getAverageFcol_team(df, dat, 'chanceCreationCrossing')
    df = getAverageFcol_team(df, dat, 'chanceCreationShooting')
    df = getAverageFcol_team(df, dat, 'defencePressure')
    df = getAverageFcol_team(df, dat, 'defenceAggression')
    df = getAverageFcol_team(df, dat, 'defenceTeamWidth')
    df = getAverageFcol_players(df, dat, 'crossing')
    df = getAverageFcol_players(df, dat, 'finishing')
    df = getAverageFcol_players(df, dat, 'heading_accuracy')
    df = getAverageFcol_players(df, dat, 'volleys')
    df = getAverageFcol_players(df, dat, 'dribbling')
    df = getAverageFcol_players(df, dat, 'curve')
    df = getAverageFcol_players(df, dat, 'long_passing')
    df = getAverageFcol_players(df, dat, 'aggression')
    df = getAverageFcol_players(df, dat, 'short_passing')
    df = getAverageFcol_players(df, dat, 'potential')
    df = getAverageFcol_players(df, dat, 'overall_rating')
    df = getAverageFcol_players(df, dat, 'long_shots')
    df = getAverageFcol_players(df, dat, 'ball_control')
    return df

def orderData():
    dat = sqlite3.connect("database.sqlite")
    tables = "season, home_team_api_id, away_team_api_id, home_team_goal, away_team_goal, home_player_1, home_player_2, home_player_3, home_player_4, home_player_5, home_player_6, home_player_7, home_player_8, home_player_9, home_player_10, home_player_11, away_player_1, away_player_2, away_player_3, away_player_4, away_player_5, away_player_6, away_player_7, away_player_8, away_player_9, away_player_10, away_player_11"
    query = dat.execute("SELECT " + tables + " From Match")
    cols = [column[0] for column in query.description]
    df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
    # df = df[350:370]
    # fromXML(newMatch)
    df = addF(df,dat)
    df = resultCol(df)
    remove_not_data_good(df)
    df = zero_handle(df)
    df = normelize_data(df)
    data = trainTest(df)
    # data[0].drop('season')
    # data[1].drop('season')
    data[0].to_excel(r'trainData1.xlsx', index=False)
    data[1].to_excel(r'TestData1.xlsx', index=False)
    return data


# orderData()
