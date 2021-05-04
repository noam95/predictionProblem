from abc import abstractmethod
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
from Preprocess import resultCol
from sklearn.feature_selection import chi2
# import matplotlib as plt
from matplotlib import pyplot as plt
import openpyxl
from sklearn.feature_selection import SelectFromModel
import os.path
from os import path


class Strategy(object):
    @abstractmethod
    def do_algorithm(self, model):
        pass

    @abstractmethod
    def get_feature_importance(self, model):
        pass

    @abstractmethod
    def analyze_model(self, model):
        pass

    def get_x_train(self):
        return self.x_train



class Context():

    def __init__(self, strategy: Strategy) -> None:
        """
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        """

        self._strategy = strategy

    @property
    def strategy(self) -> Strategy:
        """
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        """

        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        """
        Usually, the Context allows replacing a Strategy object at runtime.
        """

        self._strategy = strategy

    def do_some_business_logic(self) -> None:
        """
        The Context delegates some work to the Strategy object instead of
        implementing multiple versions of the algorithm on its own.
        """

        # ...

        result = self._strategy.do_algorithm()
        return result

    #     # ...
    #
    # def extract_to_excel(self, path, name, columns):
    #     data_df = {'Model name': name,
    #                'Number of features': len(columns),
    #                'Accurancy':self}
    #
    #     df = pd.DataFrame(columns=columns,)

def getTrainTest():
    train = pd.read_csv("trainData (1).csv")
    test = pd.read_csv("TestData (1).csv")
    return train, test


class FrameWorkRandomForest(Strategy):
    def __init__(self, model, x_train, y_train , x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.prediction = None
        self.accur = None

    def do_algorithm(self):
        self.model.fit(self.x_train, self.y_train)
        self.prediction = self.model.predict(self.x_test)
        self.accur  = accuracy_score(self.y_test, self.prediction)
        print(accuracy_score(self.y_test, self.prediction))
        f_i = self.model.feature_importances_
        # print(self.model.metrics.average_precision_score)
        return self.accur, self.model.feature_importances_

    def removeFeatures(self):
        '''
        optinal: function that remove features with zero importance
        :return:
        '''
        pass

    # def extract_to_excel(self, path, name, columns):
    #     data_df = {'Model name': name,
    #                'Number of features': len(columns),
    #                'Accurancy': self.accur
    #                }
    #
    #     df = pd.DataFrame(data_df, columns=columns)
    #     df.to_excel(path)

class FrameWorkANN(Strategy):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.prediction = None
        self.coef_ = None
        self.accur = None
    #
    # def __init__(self, model, x, y):
    #     self.model = model
    #     self.x = x
    #     self.y = y
    #     self.prediction = None
    #     self.coef_ = None
    #     self.accur = None

    def do_algorithm(self):
        x_train_new = SelectKBest(chi2, k=10).fit_transform(self.x_train, self.y_train)
        x_test_new = SelectKBest(chi2, k=10).fit_transform(self.x_test, self.y_test)
        # self.model.fit(self.x_train, self.y_train)
        # self.coef_ = self.model.coef_
        # x_train, x_test, y_train, y_test = train_test_split(X_new, self.y, test_size=0.3, random_state=42)
        self.model.fit(x_train_new, self.y_train)
        self.prediction = self.model.predict(x_test_new)
        self.accur = accuracy_score(self.y_test, self.prediction)
        print(accuracy_score(self.y_test, self.prediction))
        return self.accur

    def getFeatureImportance():
        '''
        optional: Feature inmportance
        :return:
        '''
        pass
        # f = []
        # for j in range(self.x_test.shape[1]):
        #     f_j = self.get_feature_importance(j, 100)
        #     f.append(f_j)
        # # Plot
        # plt.figure(figsize=(10, 5))
        # plt.bar(range(self.x_test.shape[1]), f, color="r", alpha=0.7)
        # plt.xticks(ticks=range(self.x_test.shape[1]))
        # plt.xlabel("Feature")
        # plt.ylabel("Importance")
        # plt.title("Feature importances (Iris data set)")
        # plt.show()
        # result = classification_report(self.y_test, self.prediction)
        # print(result)

    #
    # def get_feature_importance(self,j, n):
    #     s = accuracy_score(self.y_test, self.prediction) # baseline score
    #     total = 0.0
    #     for i in range(n):
    #         perm = np.random.permutation(range(self.x_test.shape[0]))
    #         X_test_ = self.x_test.copy()
    #         X_test_[:, j] = self.x_test[perm, j]
    #         self.prediction = self.model.predict(X_test_)
    #         s_ij = accuracy_score(self.y_test, self.prediction)
    #         total += s_ij
    #     return s - total / n

class FrameWorkSVM(Strategy):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.accur = None
        self.prediction = None

    def do_algorithm(self):
        #split with SelectBest k= num of features
        #train and split
        self.model.fit(self.x_train, self.y_train)
        self.accur = self.model.predict(self.x_test)
        print(self.accur)
        return self.accur


class FrameWorkBayesianNetworks(Strategy):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.prediction = None

    def do_algorithm(self):
        # split with SelectBest k= num of features
        # train and split
        self.model.fit(self.x_train, self.y_train)
        self.prediction = self.model.predict(self.x_test)
        prediction_binary = np.where(self.prediction > 0.8, 1, 0)
        self.prediction = prediction_binary
        #accurancy
        print(accuracy_score(self.y_test, self.prediction))



def checkRandomForest():
    # df = pd.read_csv('newMatch.csv')
    # df = resultCol(df)
    train, test = getTrainTest()
    columns = train.columns.values
    # df.drop('date', inplace=True, axis=1)
    # df.drop('shoton', inplace=True, axis=1)
    # df = df.fillna(df.mean())
    # print(df.isnull().values.any())
    # print(df.dtypes)
    data_train_np = train.to_numpy()
    data_test_np = test.to_numpy()
    # X, y = data_np.reshape(data_np.shape[0], data_np.shape[1]), data_np[:, 46]
    # y = y.astype('int')
    clf = RandomForestClassifier(random_state=0)

    x_train = train.iloc[:, 0:46]
    y_train = train.iloc[:, 46]
    x_test = test.iloc[:, 0:46]
    y_test = test.iloc[:, 46]
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model_test = Context(FrameWorkRandomForest(clf, x_train, y_train, x_test, y_test))

    accur, importance = model_test.do_some_business_logic()
    # model_test.extract_to_excel("test.csv", "model1", columns)
    if path.exists("RandomDFtest.xlsx"):
        #load the xl to df


    else:
        data_df = {'Model name': ["Random Forest"],
                   'Number of features': [len(columns)],
                   'Accurancy': [accur]
                   }
        i = 0
        for col in columns[0:46]:
            data_df[col] = importance[i]
            i += 1
        df = pd.DataFrame(data_df)
        # df = pd.DataFrame(data=data_df)
        # df.to_excel("RandomDFtest.xlsx")

def checkANN():
    # df = pd.read_csv('newMatch1.csv')
    # df = resultCol(df)
    # df.drop('date', inplace=True, axis=1)
    # df.drop('result', inplace=True, axis=1)
    # df = df.fillna(df.mean())
    train, test = getTrainTest()
    columns = train.columns.values
    x_train = train.iloc[:, 0:46]
    y_train = train.iloc[:, 46]
    x_test = test.iloc[:, 0:46]
    y_test = test.iloc[:, 46]
    # data_np = df.to_numpy()
    # X, y = data_np.reshape(data_np.shape[0], data_np.shape[1]), data_np[:, 48]
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = MLPClassifier()
    model_ann = Context(FrameWorkANN(clf,x_train,y_train,x_test,y_test))
    accurancy_model = model_ann.do_some_business_logic()


if __name__ == '__main__':
    checkANN()
    # checkRandomForest()


