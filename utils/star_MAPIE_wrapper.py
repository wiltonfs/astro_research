# # Fake model I use for MAPIE
# Felix Wilton
# 8/30/2023

from sklearn.base import BaseEstimator


class StarNetSciKitMAPIE(BaseEstimator):
    def __init__(self):
        print("StarNetSciKitMAPIE init")

    def fit(self, X, y):
        print("StarNetSciKitMAPIE fit")
        
    def predict(self, X, y=None):
        print("StarNetSciKitMAPIE predict")
        return y