# # Run Starnet with MAPIE to get prediction intervals
# Felix Wilton
# 8/30/2023
import numpy as np
from mapie.regression import MapieRegressor

import matplotlib.pyplot as plt
from utils.star_datasets import *
from utils.star_scikit import *
from utils.star_logger import *
from utils.star_plotter import *

def actual_confidence(prediction_interval, y_true):
    '''
    Calculates the actual confidence of the prediction interval by counting the number of times the true value falls within the prediction interval
    '''
    N = len(y_true)
    correct = 0
    for i in range(N):
        if y_true[i] >= prediction_interval[i, 0] or y_true[i] <= prediction_interval[i, 1]:
            correct += 1
    return correct / N

def avg_prediction_interval_width(prediction_interval):
    '''
    Calculates the average width of the prediction interval
    '''
    return np.mean(prediction_interval[:, 1] - prediction_interval[:, 0])

outDir = "intervalOutputs/"
logger = StarLogger(outDir)
plotter = StarPlotter(outDir)
logger.log("")
logger.log("## Starting Starnet-Intervals ##")

dataDir = "data/"
train_data_file = "synth_clean.h5"
test_data_files = ["synth_clean", "obs_APOGEE", "obs_GAIA"]
label_keys = ['teff', 'feh', 'logg', 'alpha']

iterss = [100, 1000, 10000, 100000]
batch_size = 16
alphas = [0.1, 0.5, 0.9]



train_dataset = SimpleSpectraDataset(data_file=dataDir + train_data_file, dataset='train')
X_train = train_dataset.__toX__()


for iters in iterss:
    # MAPIE only handles single label regression:
    for label in label_keys:
        epochs = (iters*batch_size) / len(train_dataset)
        logger.log("MAPIE regressor for label: " + label)
        logger.log(f"\titers = {iters:.0f}")
        logger.log(f"\tepochs = {epochs:.3f}")
        y_train = train_dataset.__toY__(label)
        mapie_regressor = MapieRegressor(StarNetScikit(iters, batch_size))
        logger.log("\tStarting fit...")
        fitTimer = time.time()
        mapie_regressor.fit(X_train, y_train)
        #print time in seconds with 2 decimal points
        logger.log(f"\tFinished fit in {(time.time() - fitTimer):.1f} seconds!")

        logger.log(f"\tStarting evaluation on datasets...")
        for test_data_file in test_data_files:
            logger.log(f"\t\t{test_data_file}")
            test_dataset = SimpleSpectraDataset(data_file=dataDir + test_data_file + ".h5", dataset='val')
            X_pred = test_dataset.__toX__()
            y_pred = test_dataset.__toY__(label)

            logger.log("\t\t\tTrying different alpha values...\n")
            for alpha in alphas:
                confidence = (1 - alpha) * 100.0
                logger.log(f"\t\t\talpha = {alpha:.2f}")
                logger.log(f"\t\t\tconfidence = {confidence:.1f}%")
                logger.log("\t\t\tStarting prediction...")
                predictionTimer = time.time()
                prediction, prediction_interval = mapie_regressor.predict(X_pred, alpha=alpha)  # alpha determines the prediction level
                logger.log(f"\t\t\tFinished prediction in {(time.time() - predictionTimer):.1f} seconds!")
                actual_conf = actual_confidence(prediction_interval, y_pred) * 100.0
                avg_interval_width = avg_prediction_interval_width(prediction_interval)
                logger.log(f"\t\t\tActual confidence: {actual_conf:.1f}%")
                logger.log(f"\t\t\tAverage prediction interval width: {avg_interval_width:.3f}")
                plotter.plot_scatter_losses_intervals(y_pred, prediction, prediction_interval, label, alpha, iters, test_data_file)
                logger.log("\t\t\tSaved plot to " + outDir)

                # Save prediction, and prediction interval to a file
                path = outDir + str(iters) + "_" + str(alpha) + "_" + test_data_file + "_" + label + "_prediction.txt"
                np.savetxt(path, np.c_[y_pred, prediction, prediction_interval[:, 0], prediction_interval[:, 1]], delimiter=',', header='y_true,y_predicted,prediction_interval 0,prediction_interval 1')
                logger.log("\t\t\tSaved prediction to " + path)

                # Append the iters, label, test_data_file, alpha, actual confidence, and avg interval width to the bottom of the overall_results.csv file
                path = outDir + "overall_results.csv"
                # If the file doesn't exist, create it and add the header row
                if not os.path.isfile(path):
                    with open(path, 'w') as f:
                        f.write("iters,test_data_file,label,alpha,actual_conf,avg_interval_width\n")
                # Append the results to the file
                with open(path, 'a') as f:
                    f.write(f"{iters},{test_data_file},{label},{alpha},{actual_conf},{avg_interval_width} \n")
                logger.log("\t\t\tSaved performance to " + path + "\n")


                
logger.log("## Finished Starnet-Intervals ##")
logger.log("Done!")
logger.close()




