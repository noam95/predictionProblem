import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score

def loadData(path):
    df = pd.read_csv(path)
    return df

def hundleMissingValues():
    pass


def normelize_data(df):
    x = df.values  # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df


df = loadData('match.csv')
newMatch = df[
    ['id', 'stage', 'home_team_api_id', 'away_team_api_id', 'home_team_goal','away_team_goal',
     ]].copy()
newMatch['result'] = newMatch['home_team_goal'] - newMatch['away_team_goal']
newMatch2 = df[['home_team_api_id', 'away_team_api_id']].copy()
i = 0
for x in newMatch['result']:

    if x is 0:
        np.delete(newMatch2, 1, 0)
        continue
    elif x < 0:
        newMatch2['class'][i] = 0
    elif x > 0:
        newMatch2['class'][i] = 1

    i = i+1



# for i in range(len(df)):
#     if df[i]['']
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

