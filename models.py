from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from Framework import frame_work
from Preprocess import resultCol
import numpy as np
import pandas as pd


#call preprocess
df = pd.read_csv('newMatch.csv')
df = resultCol(df)

df=df.fillna(df.mean())

#call framework
# train_table = df[df.season != '2015/2016']
# test_table = df[df.season == '2015/2016']

# train_table.drop('season', inplace=True, axis=1)
# smaller_df_train = train_table[0:10000]
data_np = df.to_numpy()
# print(data_np.data)
X,y = data_np.reshape(data_np.shape[0], data_np.shape[1]), data_np[:, 37]
# X_train, y_train = data_np_train.reshape(data_np_train.shape[0], data_np_train.shape[1]), data_np_train[:, 7]
#
#
# test_table.drop('season', inplace=True, axis=1)
# smaller_df_test = test_table[0:10000]
# data_np_test = smaller_df_test.to_numpy()
# X_test, y_test = data_np_test.reshape(data_np_test.shape[0], data_np_test.shape[1]), data_np_test[:, 7]

'''
#for svm:
clf = svm.SVC(kernel='rbf')
'''
'''
#for neural network:
clf = MLPClassifier()
'''
'''
random
'''
clf = RandomForestClassifier(random_state=0)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.95, random_state=42)
model_test = frame_work(clf, x_train, y_train, x_test, y_test)
model_test.run_model()
f_importence = model_test.feature_importanc()
model_test.matrices_classification("test")
# clf = svm.SVC(kernel='rbf')
# clf.fit(x_train, y_train)
# prediction = clf.predict(x_test)
# print(prediction)
# print(accuracy_score(y_test, prediction))
# # print("Model accurancy- " + str(91.64765)+ "%")

