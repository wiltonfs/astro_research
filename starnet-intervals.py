import numpy as np
from mapie.regression import MapieRegressor

import matplotlib.pyplot as plt
from utils.star_datasets import *
from utils.star_scikit import *
from utils.star_logger import *

outDir = "intervalOutputs/"
logger = StarLogger(outDir)
logger.log("\n\n## Starting Starnet-Intervals ##\n")


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
alphas = [0.1, 0.5, 0.9]
iterss = [100, 10000, 1000000]

X_train = train_dataset.__toX__()
X_pred = test_dataset.__toX__()
logger.log("X_train shape: " + str(X_train.shape))
logger.log("X_pred shape: " + str(X_pred.shape))

for iters in iterss:
    logger.log("\n\n\t\t\t\t $$$$$$$$$$$ iters = " + str(iters))
    for alpha in alphas:
        logger.log("\n\n\t\t\t $$$$$$$$$$$ alpha = " + str(alpha))
        for label in label_keys:
            logger.log("\n\nMAPIE regressor for label: " + label)
            y_train = train_dataset.__toY__(label)
            y_pred = test_dataset.__toY__(label)
            logger.log("\ty_train shape: " + str(y_train.shape))
            logger.log("\ty_pred shape: " + str(y_pred.shape))
            mapie_regressor = MapieRegressor(StarNetScikit(iters))
            logger.log("\tStarting fit...")
            mapie_regressor.fit(X_train, y_train)
            logger.log("\tFinished fit!")
            prediction, prediction_interval = mapie_regressor.predict(X_pred, alpha=alpha)  # alpha determines the prediction level
            logger.log("\tFinished prediction!")
            yerr = np.ones((2, len(X_pred)))
            logger.log("\tyerr shape: " + str(yerr.shape))
            yerr[0, :] = prediction.squeeze() - prediction_interval[:, 0].squeeze()  # Calculate the lower errors
            yerr[1, :] = prediction_interval[:, 1].squeeze() - prediction.squeeze()  # Calculate the upper errors
            # Create a scatter plot with error bars
            plt.figure(figsize=(10, 10))
            plt.errorbar(y_pred, prediction, yerr=yerr, fmt='o', label=label, color='cornflowerblue', ecolor='cornflowerblue')
            plt.xlabel("True Value")
            plt.ylabel("Predicted Value")
            plt.title("Scatter Plot with Error Bars - " + label)
            plt.legend()
            #plt.show()
            path = outDir + str(iters) + "_" + str(alpha) + "_" + label + "results.png"
            plt.savefig(path, facecolor='white', transparent=False, dpi=100, bbox_inches='tight', pad_inches=0.05)
    
logger.log("\n\n\n\n\n\nDone!")

'''
model_pred_intervals = np.zeros((len(test_dataset), len(label_keys), 3))
for i, mapie_regressor in enumerate(mapie_regressors):
            logger.log("Predicting MAPIE regressor for label: ", label_keys[i])
            prediction, prediction_interval = mapie_regressor.predict(X_pred, alpha=alpha)
            model_pred_intervals[:, i, 0] = prediction.squeeze()  # Predicted values
            model_pred_intervals[:, i, 1] = prediction.squeeze() - prediction_interval[:, 0].squeeze()  # Calculate the lower errors
            model_pred_intervals[:, i, 2] = prediction_interval[:, 1].squeeze() - prediction.squeeze()  # Calculate the upper errors
'''



