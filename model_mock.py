import random
class ModelMock:
    def __init__(self, predicted):
        self.predicted = predicted 
        pass

    def __call__(self, x):
        return random.sample(range(-5, 5), 5)   
        