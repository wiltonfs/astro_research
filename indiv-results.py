# # Plot and Analyze Individual Project Results
# Felix Wilton
# 7/31/2023
from utils.star_plotter import *
import h5py
import os

project = '31.7.2023_Simple_0'

label_keys = ['teff', 'feh', 'logg', 'alpha']
datasets = ['synth_clean', 'obs_GAIA', 'obs_APOGEE']

proj_dir = f'outputs/{project}'

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
    # Load 'ground truth labels' and 'predicted labels' datasets
    for dataset in datasets:
        gt_key = f'ground truth labels {dataset}'
        pred_key = f'predicted labels {dataset}'
        ground_truth_labels[dataset] = np.array(hf[gt_key], dtype=np.float32)
        model_pred_labels[dataset] = np.array(hf[pred_key], dtype=np.float32)

# Plot and save plot_train_progress
plotter.plot_train_progress(losses)

# TODO: Plot and save performance on eval sets

# Plot and save isochrones
plotter.plot_isochrones(model_pred_labels)
