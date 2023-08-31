# # Fake model I use for MAPIE
# Felix Wilton
# 8/30/2023

from sklearn.base import BaseEstimator


class StarNetSciKitMAPIE(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        pass
        
    def predict(self, X, y=None):
        return y