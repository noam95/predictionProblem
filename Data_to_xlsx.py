import pandas as pd
import sqlite3
import sklearn
import openpyxl

def init_dataFrames(path):
    dat = sqlite3.connect(path)
    #country
    query = dat.execute("SELECT * From Country")
    cols = [column[0] for column in query.description]
    country_table = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
    country_table.to_excel(r'C:\Users\User\Google Drive\\noam_document\אוניברסיטה\\2021\\סמסטר ו\הכנה לפרויקט\\tables\country.xlsx', index=False)

    query2 = dat.execute("SELECT * FROM League ")
    cols = [column[0] for column in query2.description]
    league_table = pd.DataFrame.from_records(data=query2.fetchall(), columns=cols)
    league_table.to_excel(r'C:\Users\User\Google Drive\\noam_document\אוניברסיטה\\2021\\סמסטר ו\הכנה לפרויקט\\tables\league.xlsx', index=False)

    query3 = dat.execute("SELECT * FROM Match")
    cols = [column[0] for column in query3.description]
    match_table = pd.DataFrame.from_records(data=query3.fetchall(), columns=cols)
    newMatch = match_table[['id', 'season', 'stage', 'home_team_api_id', 'away_team_api_id', 'home_team_goal', 'away_team_goal', 'goal']].copy()
    print(newMatch)
    match_table.to_excel(r'C:\Users\User\Google Drive\\noam_document\אוניברסיטה\\2021\\סמסטר ו\הכנה לפרויקט\\tables\match.xlsx', index=False)

    query4 = dat.execute("SELECT * FROM Player")
    cols = [column[0] for column in query4.description]
    player_table = pd.DataFrame.from_records(data=query4.fetchall(), columns=cols)
    player_table.to_excel(r'C:\Users\User\Google Drive\\noam_document\אוניברסיטה\\2021\\סמסטר ו\הכנה לפרויקט\\tables\player.xlsx', index=False)

    query5 = dat.execute("SELECT * FROM Player_Attributes")
    cols = [column[0] for column in query5.description]
    playerAtt_table = pd.DataFrame.from_records(data=query5.fetchall(), columns=cols)
    playerAtt_table.to_excel(r'C:\Users\User\Google Drive\\noam_document\אוניברסיטה\\2021\\סמסטר ו\הכנה לפרויקט\\tables\playerAttributes.xlsx', index=False)


    query6 = dat.execute("SELECT * FROM Team")
    cols = [column[0] for column in query6.description]
    team_table = pd.DataFrame.from_records(data=query6.fetchall(), columns=cols)
    team_table.to_excel(r'C:\Users\User\Google Drive\\noam_document\אוניברסיטה\\2021\\סמסטר ו\הכנה לפרויקט\\tables\team.xlsx', index=False)


    query7 = dat.execute("SELECT * FROM Team_Attributes")
    cols = [column[0] for column in query7.description]
    teamAtt_table = pd.DataFrame.from_records(data=query7.fetchall(), columns=cols)
    teamAtt_table.to_excel(r'C:\Users\User\Google Drive\\noam_document\אוניברסיטה\\2021\\סמסטר ו\הכנה לפרויקט\\tables\teamAttributes.xlsx', index=False)

init_dataFrames("database.sqlite")