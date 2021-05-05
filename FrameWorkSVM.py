from Context import Strategy


class FrameWorkSVM(Strategy):
    def __init__(self, model,TrainPath,TestPath,param=None):
        super().__init__(model,"SVM",TrainPath,TestPath, param)


    def do_algorithm(self):
        def do_algorithm(self):
            super().train()
            return self.accur


