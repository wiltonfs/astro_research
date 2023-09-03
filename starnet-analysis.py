# # Plot and Analyze Individual Starnet Results
# Felix Wilton
# 7/31/2023
from utils.star_plotter import *
from utils.star_logger import *
import h5py
import os
import argparse

# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Individual Results: Arguments')
    parser.add_argument('--p', type=str, default='Simple_16_100_0.01_0.0005_5_0.03_True', help='Project path to generate results for')
    parser.add_argument('--std', type=bool, default=False, help='If training progress plot should display loss std as transparent band (Adds a lot of mess to graph)')
    args = parser.parse_args()
    return args

# Parse command-line arguments
args = parse_arguments()
project = args.p
train_std = args.std

label_keys = ['teff', 'feh', 'logg', 'alpha']
datasets = ['synth_clean', 'obs_GAIA', 'obs_APOGEE']

proj_dir = f'outputs/{project}'

logger = StarLogger(proj_dir, 'visualsLog.txt')

# Save all figures in the same folder
plotter = StarPlotter(proj_dir, label_keys, datasets, saving=True)

# Load data from the HDF5 file
losses = {}
ground_truth_labels = {}
model_pred_labels = {}

with h5py.File(os.path.join(proj_dir, 'losses_predictions.h5'), 'r') as hf:
    # Load losses
    for key in hf.keys():
        losses[key] = np.array(hf[key], dtype=np.float32)
    logger.log("\tLoaded losses")
    # Load 'ground truth labels' and 'predicted labels' datasets
    for dataset in datasets:
        gt_key = f'ground truth labels {dataset}'
        pred_key = f'predicted labels {dataset}'
        ground_truth_labels[dataset] = np.array(hf[gt_key], dtype=np.float32)
        model_pred_labels[dataset] = np.array(hf[pred_key], dtype=np.float32)
        logger.log("\tLoaded predicted and ground truth labels for " + dataset)

# Plot and save plot_train_progress
plotter.plot_train_progress(losses, std=train_std)
logger.log("Plotted training progress")
# Plot and save violin plots on synthetic eval sets
plotter.plot_violin_loss(model_pred_labels, ground_truth_labels)
logger.log("Plotted violin plots of loss")
# Plot and save performance on all eval sets
plotter.plot_scatter_losses(model_pred_labels, ground_truth_labels)
logger.log("Plotted scatter plots of loss")
# Plot and save isochrones
plotter.plot_isochrones(model_pred_labels)
logger.log("Plotted isochrones")
logger.log("Done!")
