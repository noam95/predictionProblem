from abc import abstractmethod
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
from Preprocess import resultCol
import openpyxl



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
    # def extract_to_excel(self, path):
    #     df = self._strategy
    #     df = pd.DataFrame([[11, 21, 31], [12, 22, 32], [31, 32, 33]],
    #                       index=['one', 'two', 'three'], columns=['modelName', '', 'c'])



class FrameWorkRandomForest(Strategy):
    def __init__(self, model, x_train, y_train , x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.prediction = None

    def do_algorithm(self):
        self.model.fit(self.x_train, self.y_train)
        self.prediction = self.model.predict(self.x_test)
        print(accuracy_score(self.y_test, self.prediction))
        f_i = self.model.feature_importances_
        print(self.model.metrics.average_precision_score)
        return self.model.feature_importances_


    def get_x_train(self):
        return self.x_train

class FrameWorkANN(Strategy):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.prediction = None
        self.coef_ = None

    def do_algorithm(self):
        self.model.fit(self.x_train, self.y_train)
        # self.coef_ = self.model.coef_
        self.prediction = self.model.predict(self.x_test)
        print(accuracy_score(self.y_test, self.prediction))




class FrameWorkSVM(Strategy):
    def do_algorithm(self):
        pass

class FrameWorkBayesianNetworks(Strategy):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.prediction = None

    def do_algorithm(self):
        self.model.fit(self.x_train, self.y_train)
        self.prediction = self.model.predict(self.x_test)
        prediction_binary = np.where(self.prediction >0.8 , 1, 0)
        self.prediction = prediction_binary
        #accurancy
        print(accuracy_score(self.y_test, self.prediction))



def checkRandomForest():
    df = pd.read_csv('newMatch.csv')
    df = resultCol(df)
    df.drop('date', inplace=True, axis=1)
    df.drop('shoton', inplace=True, axis=1)
    df = df.fillna(df.mean())
    # print(df.isnull().values.any())
    # print(df.dtypes)
    data_np = df.to_numpy()
    X, y = data_np.reshape(data_np.shape[0], data_np.shape[1]), data_np[:, 29]
    y = y.astype('int')
    clf = RandomForestClassifier(random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model_test = Context(FrameWorkRandomForest(clf, x_train, y_train, x_test, y_test))
    model_test.do_some_business_logic()

def checkANN():
    df = pd.read_csv('newMatch.csv')
    df = resultCol(df)
    df.drop('date', inplace=True, axis=1)
    df.drop('shoton', inplace=True, axis=1)
    df = df.fillna(df.mean())
    data_np = df.to_numpy()
    X, y = data_np.reshape(data_np.shape[0], data_np.shape[1]), data_np[:, 29]
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = MLPClassifier()
    model_ann = Context(FrameWorkANN(clf,x_train,y_train,x_test,y_test))
    model_ann.do_some_business_logic()

if __name__ == '__main__':
    # checkANN()
    checkRandomForest()


