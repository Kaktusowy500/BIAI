
class ModelMock:
    def __init__(self, predicted):
        self.predicted = predicted 
        pass

    def __call__(self, x):
        return self.predicted    
        