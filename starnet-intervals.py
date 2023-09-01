import numpy as np
from mapie.regression import MapieRegressor

import matplotlib.pyplot as plt
from utils.star_datasets import *
from utils.star_scikit import *

print("\n\n## Starting Starnet-Intervals ##\n")

train_data_file = "data/synth_clean.h5"
train_dataset = SimpleSpectraDataset(data_file=train_data_file, dataset='train')
test_dataset = SimpleSpectraDataset(data_file=train_data_file, dataset='val')
label_keys = ['teff', 'feh', 'logg', 'alpha']

# MAPIE only handles single label regression:
# We will already have a model trained
# Make mapie_regressor for each label
# Pass the train data in and "train" the regressor
# Pass the test data in and "predict" the regressor
# Compile the results into one output
alpha = 0.1
X_train = train_dataset.__toX__()
X_pred = test_dataset.__toX__()
print("X_train shape: " + str(X_train.shape))
print("X_pred shape: " + str(X_pred.shape))
for label in label_keys:
    print("\n\nMAPIE regressor for label: ", label)
    y_train = train_dataset.__toY__(label)
    y_pred = test_dataset.__toY__(label)
    print("\ty_train shape: " + str(y_train.shape))
    print("\ty_pred shape: " + str(y_pred.shape))
    mapie_regressor = MapieRegressor(StarNetScikit())
    print("\tStarting fit...")
    mapie_regressor.fit(X_train, y_train)
    print("\tFinished fit!")
    prediction, prediction_interval = mapie_regressor.predict(X_pred, alpha=0.1)  # alpha determines the prediction level
    print("\tFinished prediction!")
    yerr = np.ones((2, len(X_pred)))
    print("\tyerr shape: " + str(yerr.shape))
    yerr[0, :] = prediction.squeeze() - prediction_interval[:, 0].squeeze()  # Calculate the lower errors
    yerr[1, :] = prediction_interval[:, 1].squeeze() - prediction.squeeze()  # Calculate the upper errors
    # Create a scatter plot with error bars
    plt.errorbar(y_pred, prediction, yerr=yerr, fmt='o', label=label, color='red', ecolor='red')
    plt.xlabel("True Value")
    plt.ylabel("Predicted Value")
    plt.title("Scatter Plot with Error Bars - " + label)
    plt.legend()
    plt.show()
    


'''
model_pred_intervals = np.zeros((len(test_dataset), len(label_keys), 3))
for i, mapie_regressor in enumerate(mapie_regressors):
            print("Predicting MAPIE regressor for label: ", label_keys[i])
            prediction, prediction_interval = mapie_regressor.predict(X_pred, alpha=alpha)
            model_pred_intervals[:, i, 0] = prediction.squeeze()  # Predicted values
            model_pred_intervals[:, i, 1] = prediction.squeeze() - prediction_interval[:, 0].squeeze()  # Calculate the lower errors
            model_pred_intervals[:, i, 2] = prediction_interval[:, 1].squeeze() - prediction.squeeze()  # Calculate the upper errors
'''



