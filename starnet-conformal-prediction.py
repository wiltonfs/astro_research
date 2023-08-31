import numpy as np
from mapie.regression import MapieRegressor
import matplotlib.pyplot as plt
from utils.star_scikit import *
from utils.star_datasets import *

train_data_file = "data/synth_clean.h5"
train_dataset = SimpleSpectraDataset(train_data_file, 'train')
test_dataset = SimpleSpectraDataset(train_data_file, 'val')

# MAPIE only handles single label regression:
# Make mapie_regressor for each label
# 

model = StarNetSciKit()
mapie_regressor = MapieRegressor(model)
mapie_regressor.fit(train_dataset, None)

# Predict with uncertainty using MAPIERegressor
prediction, prediction_interval = mapie_regressor.predict(test_dataset, alpha=0.1)  # alpha determines the prediction level


