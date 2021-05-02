import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import *


class frame_work:

    def __init__(self, model, x_train, y_train, x_test, y_test):

        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        # self.x_train, self.x_test,self.y_train,self.y_test = train_test_split(X,y, test_size=0.3, random_state=42)
        self.y_test = y_test
        self.x_test = x_test
        self.prediction = None


    def run_model(self):
        self.model.fit(self.x_train, self.y_train)
        self.prediction = self.model.predict(self.x_test)

    def matrices_classification(self, path):
        target_names = ['win', 'loose']
        result = classification_report(self.y, self.prediction, target_names=target_names)

    def feature_importanc(self):
        print(accuracy_score(self.y_test, self.prediction))
        return self.model.feature_importances_

    def f_importances(coef, names):
        imp = coef
        imp, names = zip(*sorted(zip(imp, names)))
        plt.barh(range(len(names)), imp, align='center')
        plt.yticks(range(len(names)), names)
        plt.show()