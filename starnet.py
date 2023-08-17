# # Reliable Neural Net Architecture for Stellar Predictions
# Felix Wilton
# 6/27/2023

import os
from datetime import date
import numpy as np
import csv
import h5py
from collections import defaultdict
import time
import torch
from torch.optim.lr_scheduler import StepLR
import argparse

from utils.star_model import *
from utils.star_logger import *
from utils.star_datasets import *
# Function to parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='StarNet Hyperparameters')
    parser.add_argument('--id', type=str, default='Simple', help='Project identifier')
    parser.add_argument('--bs', type=int, default=16, help='Batch size for training (default: 16)')
    parser.add_argument('--lrI', type=float, default=0.01, help='Initial Learning rate for training (default: 0.01)')
    parser.add_argument('--lrF', type=float, default=0.0005, help='Final Learning rate for training. Learning rate approaches this through a geometric decay (default: 0.0005)')
    parser.add_argument('--i', type=int, default=int(1e2), help='Training iterations (default: 100)')
    parser.add_argument('--vs', type=int, default=5, help='Validation steps during training (default: 5)')
    parser.add_argument('--ns', type=float, default=0.03, help='STD of noise added to spectra (default: 0.03)')
    parser.add_argument('--tl', type=bool, default=True, help='If the model is evaluated on other datasets (Setting this to False will speed up runtime)')
    args = parser.parse_args()
    return args

# Parse command-line arguments
args = parse_arguments()

## MAIN PARAMETERS
project_id = args.id
batch_size = args.bs
initial_learning_rate = args.lrI
final_learning_rate = args.lrF
total_batch_iters = args.i
val_steps = args.vs
TRANSFER_LEARNING = args.tl
# Noise parameters
noise_std = args.ns
noise_mean = 0


val_batch_size = 1024
output_dir = 'outputs'
data_dir = 'data'

label_keys = ['teff', 'feh', 'logg', 'alpha']
datasets = ['synth_clean', 'obs_GAIA', 'obs_APOGEE']
TRAIN_DATASET_SELECT = 0
project_name = f"{args.id}_{args.bs}_{args.i}_{args.lrI}_{args.lrF}_{args.vs}_{args.ns}_{TRANSFER_LEARNING}"
project_dir = os.path.join(output_dir, project_name)

if os.path.exists(project_dir):
    # Log a warning that the project folder already exists and will be overwritten
    print(f"WARNING: Project folder '{project_name}' already exists. Overwriting...")

# Create or overwrite the project folder
os.makedirs(project_dir, exist_ok=True)

logger = StarLogger(project_dir)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
logger.log('Using Torch version: %s' % (torch.__version__))
logger.log('Using a %s device\n' % (device))
logger.log(f"Folder created: {project_dir}")

# Record model parameters
params = {
    'project_id': project_id,
    'batch_size': batch_size,
    'initial_learning_rate': initial_learning_rate,
    'final_learning_rate': final_learning_rate,
    'total_batch_iters': total_batch_iters,
    'val_steps': val_steps,
    'TRANSFER_LEARNING': TRANSFER_LEARNING,
    'noise_mean': noise_mean,
    'noise_std': noise_std,
    'label_keys': label_keys,
    'datasets': datasets,
    'TRAIN_DATASET_SELECT': TRAIN_DATASET_SELECT
}
output_file = os.path.join(project_dir, 'params.txt')
with open(output_file, 'w') as paramFile:
    for key, value in params.items():
        paramFile.write(f"{key} = {value}\n")
logger.log('Logged model parameters')

train_data_file = os.path.join(data_dir, datasets[TRAIN_DATASET_SELECT] + '.h5')
model_filename =  os.path.join(project_dir,'model.pth.tar')

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

# Create the validation datasets
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
optimizer = torch.optim.Adam(model.parameters(), initial_learning_rate, weight_decay=0)

# Calculate the learning rate schedule
lr_factor = (final_learning_rate / initial_learning_rate) ** (1.0 / total_batch_iters)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_factor ** epoch)


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
        scheduler.step()
        # Save losses to find average later
        running_loss.append(float(loss))

        cur_iter += 1
        
        # Display progress
        if cur_iter % verbose_iters == 0:
            logger.log('[Iter %i, %0.0f%%]' % (cur_iter, cur_iter/total_batch_iters*100))
            val_start_time = time.time()
            # Take average and std of training losses
            train_loss = np.nanmean(running_loss)
            train_std = np.nanstd(running_loss)
            losses['train_loss'].append(train_loss)
            losses['train_std'].append(train_std)
            losses['iter'].append(cur_iter)
            std_perc = train_std / train_loss * 100
            logger.log(f'\tTrain Loss: {train_loss:0.4f} +- {std_perc:0.2f}%')
            logger.log('\tTrain time taken: %0.0f seconds' % (val_start_time - iters_start_time))

            # Set parameters to not trainable
            model.eval()

            # Evaluate on validation set with noise and display losses
            with torch.no_grad():
                running_loss = []
                for val_batch in val_dataloaders[datasets[TRAIN_DATASET_SELECT]]:
                    val_batch['spectrum'] += torch.randn_like(val_batch['spectrum']) * noise_std + noise_mean
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
                val_std = np.nanstd(running_loss)
                losses['val_loss'].append(val_loss)  
                losses['val_std'].append(val_std)  
                std_perc = val_std / val_loss * 100
                logger.log(f'\tVal Loss: {val_loss:0.4f} +- {std_perc:0.2f}%')
                logger.log('\tValidation time taken: %0.0f seconds' % (time.time() - val_start_time))

                eval_start_time = time.time()

                # Evaluate on transfer learning evaluation sets and display losses
                if TRANSFER_LEARNING:
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
                        eval_loss = np.nanmean(running_loss)
                        eval_std = np.nanstd(running_loss)
                        losses['eval_loss_'+dataset].append(eval_loss)         
                        losses['eval_std_'+dataset].append(eval_std)
                    logger.log('\tTransfer learning evaluation time taken: %0.0f seconds' % (time.time() - eval_start_time))  
            
            running_loss = []
            

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
            train_time = time.time() - train_start_time
            logger.log('Total training time: %0.0f seconds' % (train_time))
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
with h5py.File(os.path.join(project_dir, 'losses_predictions.h5'), 'w') as hf:
    # Save losses
    for key in losses.keys():
        hf.create_dataset(key, data=np.array(losses[key], dtype=np.float32))
    # Save 'ground truth labels' and 'predicted labels' datasets separately for each dataset
    for dataset in datasets:
        gt_key = f'ground truth labels {dataset}'
        pred_key = f'predicted labels {dataset}'
        hf.create_dataset(gt_key, data=ground_truth_labels[dataset], dtype=np.float32)
        hf.create_dataset(pred_key, data=model_pred_labels[dataset], dtype=np.float32)

logger.log("Saved losses and evaluation predictions")

# Save results to results file
results_dir = os.path.join(output_dir, 'results.csv')
mode = 'a' if os.path.exists(results_dir) else 'w'
field_names = ['project name', 'iters', 'epochs', 'train time (s)', 'batch size', 'initial learning rate', 'final learning rate', 'noise std',
               'val_loss', 'val_loss_std', 'eval_loss_synth_clean', 'eval_std_synth_clean',
               'eval_loss_obs_GAIA', 'eval_std_obs_GAIA', 'eval_loss_obs_APOGEE', 'eval_std_obs_APOGEE']
# Open the CSV file in append mode or create a new one
with open(results_dir, mode, newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=field_names)

    # If it's a new file, write the header row
    if mode == 'w':
        writer.writeheader()

    # Write a new row for the current experiment results
    result_row = {
        'project name': project_name,
        'iters': cur_iter,
        'epochs':epochs,
        'train time (s)': round(train_time),
        'batch size': batch_size,
        'initial learning rate': initial_learning_rate,
        'final learning rate': final_learning_rate,
        'noise std': noise_std,
        'val_loss': losses['val_loss'][-1],
        'val_loss_std': losses['val_std'][-1],
    }

    # Conditionally add transfer learning evaluation results if TRANSFER_LEARNING is True
    if TRANSFER_LEARNING:
        result_row['eval_loss_synth_clean'] = losses['eval_loss_synth_clean'][-1]
        result_row['eval_std_synth_clean'] = losses['eval_std_synth_clean'][-1]
        result_row['eval_loss_obs_GAIA'] = losses['eval_loss_obs_GAIA'][-1]
        result_row['eval_std_obs_GAIA'] = losses['eval_std_obs_GAIA'][-1]
        result_row['eval_loss_obs_APOGEE'] = losses['eval_loss_obs_APOGEE'][-1]
        result_row['eval_std_obs_APOGEE'] = losses['eval_std_obs_APOGEE'][-1]
    else:
        result_row['eval_loss_synth_clean'] = 'N/A'
        result_row['eval_std_synth_clean'] = 'N/A'
        result_row['eval_loss_obs_GAIA'] = 'N/A'
        result_row['eval_std_obs_GAIA'] = 'N/A'
        result_row['eval_loss_obs_APOGEE'] = 'N/A'
        result_row['eval_std_obs_APOGEE'] = 'N/A'

    # Write the row to the CSV file
    writer.writerow(result_row)


logger.log("Results have been saved to " + results_dir)


logger.log("Done!")
logger.close()

#Print project name as the output of the script
print(project_name)



