from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from Framework import frame_work
from Preprocess import resultCol
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import svm

#call preprocess
df = pd.read_csv('newMatch1.csv')
df = resultCol(df)
df = df.fillna(df.mean())


def round2(ind):
    data_np = df.to_numpy()
    data_np_shrink = data_np[:, ind]
    X, y = data_np_shrink.reshape(data_np.shape[0], data_np.shape[1]), data_np[:, 48]
    y = y.astype('int')
    rand = RandomForestClassifier(random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
    model_test = frame_work(rand, x_train, y_train, x_test, y_test)
    model_test.run_model()  # run the commands: fit+predict
    f_importance = model_test.feature_importance()

#call framework
# train_table = df[df.season != '2015/2016']
# test_table = df[df.season == '2015/2016']
# train_table.drop('season', inplace=True, axis=1)
# smaller_df_train = train_table[0:10000]
data_np = df.to_numpy()
# print(data_np.data)
X,y = data_np.reshape(data_np.shape[0], data_np.shape[1]), data_np[:, 48]
y = y.astype('int')
# X_train, y_train = data_np_train.reshape(data_np_train.shape[0], data_np_train.shape[1]), data_np_train[:, 7]
#
#
# test_table.drop('season', inplace=True, axis=1)
# smaller_df_test = test_table[0:10000]
# data_np_test = smaller_df_test.to_numpy()
# X_test, y_test = data_np_test.reshape(data_np_test.shape[0], data_np_test.shape[1]), data_np_test[:, 7]

'''
for svm:
clf = svm.SVC(kernel='rbf')
'''
# clf = svm.SVC(kernel='rbf')
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# model_test = frame_work(clf, x_train, y_train, x_test, y_test)
# model_test.run_model()  # run the commands: fit+predict
# f_importance = model_test.feature_importance()
# model_test.precision()


# # print("Model accurancy- " + str(91.64765)+ "%")

'''
for neural network:
clf = MLPClassifier()
'''


'''
for random forest:
'''
rand = RandomForestClassifier(random_state=0)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
model_test = frame_work(rand, x_train, y_train, x_test, y_test)
model_test.run_model()  # run the commands: fit+predict
f_importance = model_test.feature_importance()
ind = np.argpartition(f_importance, -10)[-10:]
round2(ind)
ten_importance = f_importance[ind]
model_test.matrices_classification("test")

'''
for Bayesian Ridge Regression:
'''
# reg = linear_model.BayesianRidge()
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# model_test = frame_work(reg, x_train, y_train, x_test, y_test)
# model_test.run_model()  # run the commands: fit+predict
# f_importance = model_test.feature_importance()








