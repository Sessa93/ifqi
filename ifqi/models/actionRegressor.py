import numpy as np
from copy import deepcopy


class ActionRegressor(object):
    def __init__(self, model, discreteActions, **params):
        self.actions = np.unique(discreteActions)
        self.models = self.initModel(model, **params)

    def fit(self, X, y, **kwargs):
        for i in range(len(self.models)):
            action = self.actions[i]
            idxs = np.argwhere(X[:, -1] == action).ravel()
            self.models[i].fit(X[idxs, :-1], y[idxs], **kwargs)

    def predict(self, x, **kwargs):
        action = x[0, -1]
        idxs = np.asscalar(np.argwhere(self.actions == action))
        output = self.models[idxs].predict(x[:, :-1], **kwargs)

        return output

    def adapt(self, iteration):
        for i in range(len(self.models)):
            self.models[i].adapt(iteration)

    def initModel(self, model, **params):
        models = list()
        for i in range(len(self.actions)):
            models.append(model(**params))
        return models
