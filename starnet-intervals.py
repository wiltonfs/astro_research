import numpy as np
from mapie.regression import MapieRegressor

import matplotlib.pyplot as plt
from utils.star_datasets import *
from utils.star_scikit import *
from utils.star_logger import *

outDir = "intervalOutputs/"
logger = StarLogger(outDir)
logger.log("")
logger.log("## Starting Starnet-Intervals ##")


train_data_file = "data/synth_clean.h5"
train_dataset = SimpleSpectraDataset(data_file=train_data_file, dataset='train')
test_dataset = SimpleSpectraDataset(data_file=train_data_file, dataset='val')
label_keys = ['teff']#label_keys = ['teff', 'feh', 'logg', 'alpha']

# MAPIE only handles single label regression:
# We will already have a model trained
# Make mapie_regressor for each label
# Pass the train data in and "train" the regressor
# Pass the test data in and "predict" the regressor
# Compile the results into one output
alphas = [0.1, 0.5, 0.9]
batch_size = 16
iterss = [100, 1000, 10000, 100000, 1000000]

X_train = train_dataset.__toX__()
X_pred = test_dataset.__toX__()
logger.log("X_train shape: " + str(X_train.shape))
logger.log("X_pred shape: " + str(X_pred.shape) + "\n")

for iters in iterss:
    for label in label_keys:
        epochs = (iters*batch_size) / len(train_dataset)
        logger.log("MAPIE regressor for label: " + label)
        logger.log(f"\titers = {iters:.0f}")
        logger.log(f"\tepochs = {epochs:.3f}")
        y_train = train_dataset.__toY__(label)
        y_pred = test_dataset.__toY__(label)
        mapie_regressor = MapieRegressor(StarNetScikit(iters, batch_size))
        logger.log("\tStarting fit...")
        fitTimer = time.time()
        mapie_regressor.fit(X_train, y_train)
        #print time in seconds with 2 decimal points
        logger.log(f"\tFinished fit in {(time.time() - fitTimer):.1f} seconds!")
        logger.log("\tTrying different alpha values...\n")
        for alpha in alphas:
            confidence = (1 - alpha) * 100.0
            logger.log(f"\t\talpha = {alpha:.2f}")
            logger.log(f"\t\tconfidence = {confidence:.1f}%")
            logger.log("\t\tStarting prediction...")
            predictionTimer = time.time()
            prediction, prediction_interval = mapie_regressor.predict(X_pred, alpha=alpha)  # alpha determines the prediction level
            logger.log(f"\t\tFinished prediction in {(time.time() - predictionTimer):.1f} seconds!")
            yerr = np.ones((2, len(X_pred)))
            yerr[0, :] = prediction.squeeze() - prediction_interval[:, 0].squeeze()  # Calculate the lower errors
            yerr[1, :] = prediction_interval[:, 1].squeeze() - prediction.squeeze()  # Calculate the upper errors
            yerr = np.abs(yerr)
            # Create a scatter plot with error bars
            plt.figure(figsize=(10, 10))
            plt.errorbar(y_pred, prediction, yerr=yerr, fmt='o', label=label, color='cornflowerblue', ecolor='cornflowerblue')
            # Add a line for perfect correlation
            plt.plot(y_pred, y_pred, color='black', label='Perfect Correlation', alpha=0.5)
            plt.xlabel("True Value")
            plt.ylabel("Predicted Value")
            # title should indicate confidence
            plt.title(f"Scatter Plot with Error Bars Indicating {confidence:.1f}% Confidence - " + label)
            plt.legend()
            #plt.show()
            path = outDir + str(iters) + "_" + str(alpha) + "_" + label + "_results.png"
            plt.savefig(path, facecolor='white', transparent=False, dpi=100, bbox_inches='tight', pad_inches=0.05)
            logger.log("\t\tSaved plot to " + path + "\n")
    
logger.log("\n\nDone!")
logger.close()




