from abc import abstractmethod
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from os import path


class Strategy(object):

    def __init__(self, model,ModelName,TrainPath,TestPath, param =None):
        self.model = model
        self.x_train, self.y_train, self.x_test, self.y_test = self.get_x_train(TrainPath,TestPath)
        self.prediction = None
        self.accur = None
        self.param = param
        self.ModelName = ModelName

    '''
    this function uses the 2 method from sklearn - fit and prediction to train the model and predict test part
    '''
    @abstractmethod
    def train(self):
        self.model.fit(self.x_train, self.y_train)
        self.prediction = self.model.predict(self.x_test)
        self.accur = accuracy_score(self.y_test, self.prediction)

    @abstractmethod
    def get_feature_importance(self, model):
        pass
    @abstractmethod
    def plot_feature_importance(importance, names, model_type):
        pass

    @abstractmethod
    def analyze_model(self, model):
        pass
    '''
    this function split features columns and labels column
    '''
    def get_x_train(self,TrainPath,TestPath):
        train = pd.read_csv(TrainPath)
        test = pd.read_csv(TestPath)
        Fnum = len(train.columns)
        x_train = train.iloc[:, 0:Fnum - 1]
        y_train = train.iloc[:, Fnum - 1]
        x_test = test.iloc[:, 0:Fnum - 1]
        y_test = test.iloc[:, Fnum - 1]
        return x_train, y_train, x_test, y_test

    def grid_search(self):
        model_GS = GridSearchCV(self.model, self.param)
        self.model = model_GS
    '''
    this function get data and file name to load all results :
    measures, num of features
    '''
    def insertDataToCSV(self, df,name="1"):
        file = "plots/"+ self.ModelName +"_"+name+".csv"
        if path.exists(file):
            data_to_load = pd.read_csv(file)
            data_to_load = pd.DataFrame(data_to_load)
            # adding to exist the new row
            data_to_load = pd.concat([data_to_load, df], axis=0, ignore_index=True)
            # data_to_load.loc[len(data_to_load.index)] = df  # op1
            # data_to_load.append(df)  # op2
            data_to_load.to_csv(file, index=False)
        else:  # not exist yet
            df.to_csv(file, index=False)

    @abstractmethod
    def getCsvData(self):
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

    def run_model(self) -> None:
        """
        The Context delegates some work to the Strategy object instead of
        implementing multiple versions of the algorithm on its own.
        """

        # ...

        result = self._strategy.train()
        return result








