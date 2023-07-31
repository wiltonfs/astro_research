# # Reliable Neural Net Architecture for Stellar Predictions
# Felix Wilton
# 6/27/2023

# TODO summary file, noise generation, synth_noised

import os
from datetime import date
import numpy as np
import h5py
from collections import defaultdict
import time
import torch

from utils.star_model import *
from utils.star_logger import *
from utils.star_datasets import *

## MAIN PARAMETERS
project_id = "Testing"
batch_size = 16
val_batch_size = 1024
learning_rate = 0.001
total_batch_iters = int(1e2)
val_steps = 5
output_dir = 'outputs'
data_dir = 'data'

# Noise parameters
noise_mean = 0
noise_std = 0.03

label_keys = ['teff', 'feh', 'logg', 'alpha']
datasets = ['synth_clean', 'synth_noised', 'obs_GAIA', 'obs_APOGEE']
TRAIN_DATASET_SELECT = 0

project_name = f"{date.today().day}.{date.today().month}.{date.today().year}_{project_id}_"
# Check if the folder already exists and increment the number if needed
count = 0
while os.path.exists(os.path.join('outputs', project_name + str(count))):
    count += 1
project_name += str(count)

output_dir = os.path.join(output_dir, project_name)
os.makedirs(output_dir)

logger = StarLogger(output_dir)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger.log('Using Torch version: %s' % (torch.__version__))
logger.log('Using a %s device\n' % (device))
logger.log(f"Folder created: {output_dir}")

# Record model parameters
params = {
    'project_id': project_id,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'total_batch_iters': total_batch_iters,
    'val_steps': val_steps,
    'noise_mean': noise_mean,
    'noise_std': noise_std,
    'label_keys': label_keys,
    'datasets': datasets,
    'TRAIN_DATASET_SELECT': TRAIN_DATASET_SELECT
}
output_file = os.path.join(output_dir, 'params.txt')
with open(output_file, 'w') as paramFile:
    for key, value in params.items():
        paramFile.write(f"{key} = {value}\n")
logger.log('Logged model parameters')

train_data_file = os.path.join(data_dir, datasets[TRAIN_DATASET_SELECT] + '.h5')
model_filename =  os.path.join(output_dir,'model.pth.tar')

# Collect mean and std of the training data for normalization
with h5py.File(train_data_file, "r") as f:
    labels_mean = [np.nanmean(f[k + ' train'][:]) for k in label_keys]
    labels_std = [np.nanstd(f[k + ' train'][:]) for k in label_keys]
    spectra_mean = np.nanmean(f['spectra train'][:]) + noise_mean
    spectra_std = np.nanstd(f['spectra train'][:]) + noise_std

## DATASETS

# Training data
train_dataset = SimpleSpectraDataset(train_data_file, 'train', label_keys)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

# Create the 4 validation datasets
val_datasets = {}
val_dataloaders = {}
for dataset in datasets:
    load_path = os.path.join(data_dir, dataset+'.h5')
    val_datasets[dataset] = SimpleSpectraDataset(load_path, 'val', label_keys)
    val_dataloaders[dataset] = torch.utils.data.DataLoader(val_datasets[dataset], batch_size=val_batch_size, shuffle=False, num_workers=0, pin_memory=True)
    logger.log("Created validation dataset for " + dataset + " with size " + str(len(val_datasets[dataset])))

logger.log('The training set consists of %i spectra.' % (len(train_dataset)))
epochs = (total_batch_iters*batch_size) / len(train_dataset)
logger.log(f'At {total_batch_iters} iterations, with {batch_size} samples per batch, this model will "see" {epochs:.2f} epochs')

model = StarNet(label_keys, device, train_dataset, spectra_mean, spectra_std, labels_mean, labels_std)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=0)

## TRAIN MODEL
cur_iter = 0
verbose_iters = round(total_batch_iters/val_steps)
losses = defaultdict(list)
running_loss = []

logger.log('Started Training...')

# Start timer
train_start_time = time.time()
iters_start_time = time.time()
val_start_time = time.time()
save_start_time = time.time()

# Continuously loop over the training set
while cur_iter < (total_batch_iters):
    for train_batch in train_dataloader:
        # Add noise to train_batch
        train_batch['spectrum'] += torch.randn_like(train_batch['spectrum']) * noise_std + noise_mean
         # Set parameters to trainable
        model.train()
        # Switch to GPU if available
        train_batch = batch_to_device(train_batch, device)
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward propagation
        label_preds = model(train_batch['spectrum'], norm_in=True, denorm_out=False)
        # Compute mean-squared-error loss between predictions and normalized targets
        loss = torch.nn.MSELoss()(label_preds, model.normalize(train_batch['labels'], model.labels_mean, model.labels_std))
        # Back-propagation
        loss.backward()
        # Weight updates
        optimizer.step()
        # Save losses to find average later
        running_loss.append(float(loss))

        cur_iter += 1
        
        # Display progress
        if cur_iter % verbose_iters == 0:
            logger.log('[Iter %i, %0.0f%%]' % (cur_iter, cur_iter/total_batch_iters*100))
            val_start_time = time.time()
            # Take average of training losses
            train_loss = np.nanmean(running_loss)
            losses['train_loss'].append(train_loss)
            losses['iter'].append(cur_iter)
            logger.log('\tTrain Loss: %0.4f' % (train_loss))
            logger.log('\tTrain time taken: %0.0f seconds' % (val_start_time - iters_start_time))

            # Set parameters to not trainable
            model.eval()
            
            # Evaluate on validation set and display losses
            with torch.no_grad():
                for dataset in datasets:
                    running_loss = []
                    for val_batch in val_dataloaders[dataset]:
                        # Switch to GPU if available
                        val_batch = batch_to_device(val_batch, device)
                        # Forward propagation
                        label_preds = model(val_batch['spectrum'], norm_in=True, denorm_out=False)
                        # Compute mean-squared-error loss between predictions and normalized targets
                        loss = torch.nn.MSELoss()(label_preds, model.normalize(val_batch['labels'], model.labels_mean, model.labels_std))
                        # Save losses to find average later
                        running_loss.append(float(loss))
                    # Average validation loss
                    val_loss = np.nanmean(running_loss)
                    losses['val_loss_'+dataset].append(val_loss)              
            
            running_loss = []

            val_loss = losses['val_loss_'+datasets[TRAIN_DATASET_SELECT]][-1]
            logger.log('\tVal Loss: %0.4f' % (val_loss))
            logger.log('\tValidation time taken: %0.0f seconds' % (time.time() - val_start_time))

            save_start_time = time.time()

            # Save model
            torch.save({'optimizer' : optimizer.state_dict(),
                        'model' : model.state_dict(), 
                        'batch_iters' : cur_iter,
                        'losses' : losses,
                        'train_time' : time.time() - train_start_time},
                       model_filename)
            
            logger.log('\tSaving model time taken: %0.0f seconds' % (time.time() - save_start_time))
            
            iters_start_time = time.time()
            
        if cur_iter >= total_batch_iters:
            # Save model
            torch.save({'optimizer' : optimizer.state_dict(),
                        'model' : model.state_dict(), 
                        'batch_iters' : cur_iter,
                        'losses' : losses,
                        'train_time' : time.time() - train_start_time},
                       model_filename)

            logger.log('Finished Training')
            logger.log('Total training time: %0.0f seconds' % (time.time() - train_start_time))
            break

## Try model on the 4 validation datasets
model.eval() # Set parameters to not trainable
ground_truth_labels = {}
model_pred_labels = {}

logger.log("Started predicting labels")
with torch.no_grad():
    for dataset in datasets:
        ground_truth_labels[dataset] = []
        model_pred_labels[dataset] = []
            
        current_dataloader = val_dataloaders[dataset]
        for val_batch in current_dataloader:
            # Switch to GPU if available
            val_batch = batch_to_device(val_batch, device)
            # Forward propagation (and denormalize outputs)
            label_preds = model(val_batch['spectrum'], norm_in=True, denorm_out=True)
            # Save batch data for comparisons
            ground_truth_labels[dataset].append(val_batch['labels'].cpu().data.numpy())
            model_pred_labels[dataset].append(label_preds.cpu().data.numpy())
        
        ground_truth_labels[dataset] = np.concatenate(ground_truth_labels[dataset])
        model_pred_labels[dataset] = np.concatenate(model_pred_labels[dataset])
        logger.log("\tPredicted labels for " + dataset)

# Save losses and predictions on 4 data sets
with h5py.File(os.path.join(output_dir, 'losses_predictions.h5'), 'w') as hf:
    # Save 'train_loss' and 'iter' datasets separately
    hf.create_dataset('train_loss', data=np.array(losses['train_loss'], dtype=np.float32))
    hf.create_dataset('iter', data=np.array(losses['iter'], dtype=np.int32))

    # Save each 'val_loss_datasetX' separately
    for dataset_idx, dataset in enumerate(datasets):
        val_loss_key = f'val_loss_{dataset}'
        hf.create_dataset(val_loss_key, data=np.array(losses[val_loss_key], dtype=np.float32))

    # Save 'ground truth labels' and 'predicted labels' datasets separately for each dataset
    for dataset in datasets:
        gt_key = f'ground truth labels {dataset}'
        pred_key = f'predicted labels {dataset}'
        hf.create_dataset(gt_key, data=ground_truth_labels[dataset], dtype=np.float32)
        hf.create_dataset(pred_key, data=model_pred_labels[dataset], dtype=np.float32)

logger.log("Saved losses and validation predictions")

logger.log("Done!")

logger.close()



