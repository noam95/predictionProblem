import sqlite3
import xml.etree.ElementTree as ET
import pandas as pd


def hundleMissingValues(df,F):
    '''
    handle data missing values
    replace by avg feature

    :param df: data frame to removehandle missing values in
    :param F: feature to deal with
    :return:
    '''
    x = df[df[F] != 0]
    avg = x[F].mean()
    df[F].replace({0: avg}, inplace=True)
    return df

def zero_handle(df):
    '''
    handle zeros in the data

    :param df: data to work on
    :return: data with no zeros
    '''
    col_with_zero = ['ball_control', 'long_shots', 'crossing', 'finishing', 'heading_accuracy', 'volleys', 'dribbling',
                     'curve', 'long_passing', 'aggression', 'short_passing', 'potential', 'overall_rating',
                     'long_shots',
                     'ball_control']
    for col in col_with_zero:
        hundleMissingValues(df, col)
    return df

def normelize_data(df):
    '''
    normelizing the data
    exept spacific cols using for indicates(season)
    :param df: data to work on
    :return: data
    '''
    # result = df.copy()
    for feature_name in df.columns:
        if feature_name not in ['home_team_api_id', 'away_team_api_id', 'class_res','season']:
            max_value = df[feature_name].max()
            min_value = df[feature_name].min()
            df[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return df

def binning(col, cut_points, labels=None):
    '''
    handle results line
    :param col: data of all results
    :param cut_points: values of bins
    :param labels: the values to be define: 0- loose, teco, 1- win
    :return:
    '''
    minval = col.min()
    maxval = col.max()
    break_point = [minval] + cut_points + [maxval]
    if not labels:
        labels = range(len(cut_points)+1)
    colBin = pd.cut(col, bins=break_point, labels=labels, include_lowest=True)
    return colBin

def getPlayerAVGF(df,playerApiId ,colName):
    '''

    :param df: data to work on
    :param playerApiId: player to get on the avarge of a feature
    :param colName: feature name
    :return: avarage player measure for feature
    '''
    df = df[df['player_api_id'] == playerApiId]
    mean = df[colName].mean()

    if str(mean) == 'nan':
        return 0
    return mean

def getTeamAVGF_player(palyersDF, colName, matchRow, teamType):
    '''
    calc the team avarage score on spacific feature
    :param palyersDF: players measures table
    :param colName: feature name
    :param matchRow: row from match table
    :param teamType: away/home
    :return: avarage score for a team
    '''
    sumTeamF = 0
    sum_player=0
    for i in range(11):
        x = teamType+ '_player_' + str(i+1)
        player_id = matchRow[1][x]
        if str(player_id) == 'nan':
            continue
        sum_player += 1
        sumTeamF += getPlayerAVGF(palyersDF, player_id, colName)
    if sumTeamF != 0:
        return sumTeamF/sum_player
    return 0

def getAverageFcol_players(matchData,dat, F):
    '''
    calc the avg feature score for all the data- adding feature
    :param matchData: match data table- to add a feature to
    :param dat: connection to the sql database
    :param F: feature to be added
    :return: data frame with the feature added
    '''
    listData=[]
    query0 = "SELECT * FROM Player_Attributes "
    query = dat.execute(query0)
    cols = [column[0] for column in query.description]
    palyersDF = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)

    for row in matchData.iterrows():
        home = getTeamAVGF_player(palyersDF, F, row, 'home')
        away = getTeamAVGF_player(palyersDF, F, row, 'away')
        listData.append(home-away)
    matchData[F] = listData
    return matchData

def getTeamF(df, F, team_api_id):
    '''
    calc team avg Feature data
    :param df: table to get data from
    :param F: feature to be added
    :param team_api_id: team to get score to
    :return: avg of a feature for a team
    '''
    df = df[df['team_api_id'] == team_api_id]
    mean = df[F].mean()

    if str(mean) =='nan':
        return 0
    return mean

def getAverageFcol_team(df,dat, F):
    '''
    adding team feature to the data frame
    :param df: data to add a feature to
    :param dat: connection to the data base sql
    :param F: feature to be added
    :return: data frame with the feature col
    '''
    listData=[]
    query0 = "SELECT * FROM Team_Attributes "
    query = dat.execute(query0)
    cols = [column[0] for column in query.description]
    teamDF = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
    for row in df.iterrows():
        home = getTeamF(teamDF, F, row[1]['home_team_api_id'])
        away = getTeamF(teamDF, F, row[1]['away_team_api_id'])
        listData.append(home-away)
    df[F]= listData
    return df

def resultCol(df):
    '''
    get rude from teco cols
    :param df: data to work on
    :return: rows with no teco score
    '''
    df['result'] = df['home_team_goal'] - df['away_team_goal']
    bins= [-0.5, 0.5]
    group_names = ['0', 'teco', '1']
    df["class_res"] = binning(df['result'], bins, group_names)
    df = df[df.class_res != 'teco']
    return df

def remove_not_data_good(df):
    '''
    remove data used to calcs- get data ready to predict
    :param df: data to remove from
    :return: date to predict
    '''
    list_to_drop = ['home_player_1', 'home_player_2', 'home_player_3', 'home_player_4','home_player_5','home_player_6','home_player_7',
                    'home_player_8', 'home_player_9', 'home_player_10', 'home_player_11', 'away_player_1', 'away_player_2', 'away_player_3', 'away_player_4', 'away_player_5',
                    'away_player_6', 'away_player_7', 'away_player_8','away_player_9', 'away_player_10', 'away_player_11', 'result']
    df.drop(list_to_drop, inplace=True, axis=1)

def trainTest(df):
    '''
    split data train & test
    :param df: data to remove from
    :return: two data frames- train/test
    '''
    train_table = df[df.season != '2015/2016']
    test_table = df[df.season == '2015/2016']
    return train_table, test_table

def addF(df,dat):
    '''
    features to add to the data frame
    :param df: data frame to work on
    :param dat: connection to the sql database
    :return: data with features
    '''
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
    '''
    main function using to get the data procces in order to predict on
    :return: data reaty to predict
    '''
    dat = sqlite3.connect("database.sqlite")
    tables = "season, home_team_api_id, away_team_api_id, home_team_goal, away_team_goal, home_player_1, home_player_2, home_player_3, home_player_4, home_player_5, home_player_6, home_player_7, home_player_8, home_player_9, home_player_10, home_player_11, away_player_1, away_player_2, away_player_3, away_player_4, away_player_5, away_player_6, away_player_7, away_player_8, away_player_9, away_player_10, away_player_11"
    query = dat.execute("SELECT " + tables + " From Match")
    cols = [column[0] for column in query.description]
    df = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
    # df = df[350:370]
    df = addF(df,dat)
    df = resultCol(df)
    remove_not_data_good(df)
    df = zero_handle(df)
    df = normelize_data(df)
    data = trainTest(df)
    data[0].to_excel(r'trainData1.xlsx', index=False)
    data[1].to_excel(r'TestData1.xlsx', index=False)
    return data


# orderData()
