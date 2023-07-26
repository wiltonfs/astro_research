# # Reliable Neural Net Architecture for Stellar Predictions
# Felix Wilton
# 6/27/2023

# TODO summary file, noise generation

import os
from datetime import date
import numpy as np
import h5py
from collections import defaultdict
import time
import torch



from star_model import *
from star_plotter import *
from star_logger import *
from star_datasets import *

## MAIN PARAMETERS
project_id = "Testing"
batch_size = 16
learning_rate = 0.001
total_batch_iters = int(1e2)
val_steps = 5
output_dir = 'outputs'
data_dir = 'data'

# Noise parameters
ADD_NOISE = False
mean = 0
std = 0.03


label_keys = ['teff', 'feh', 'logg', 'alpha']
datasets = ['synth_clean', 'synth_noised', 'obs_GAIA', 'obs_APOGEE']
TRAIN_DATASET_SELECT = 0

project_name = f"{date.today().year}.{date.today().month}.{date.today().day}_{project_id}_0"
# Check if the folder already exists and increment the number if needed
count = 1
while os.path.exists(os.path.join('outputs', project_name)):
    project_name = f"{date.today().year}.{date.today().month}.{date.today().day}_{project_id}_{count}"
    count += 1
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
    'ADD_NOISE': ADD_NOISE,
    'mean': mean,
    'std': std,
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
    spectra_mean = np.nanmean(f['spectra train'][:]) 
    spectra_std = np.nanstd(f['spectra train'][:])



## DATASETS

# Training data
train_dataset = SimpleSpectraDataset(train_data_file, 'train', label_keys)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

# Create the 4 validation datasets
val_batch_size = 1024
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
verbose_iters = total_batch_iters/val_steps
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

        if (cur_iter < 3): #Print out the first few iterations
            logger.log(f"Started iteration {cur_iter}")

        # Add noise to train_batch
        if (ADD_NOISE == True):
            train_batch['spectrum'] += torch.randn_like(train_batch['spectrum']) * std + mean
        
        if (cur_iter < 3): #Print out the first few iterations
            logger.log(f"\tAdded noise")

        # Set parameters to trainable
        model.train()

        if (cur_iter < 3): #Print out the first few iterations
            logger.log(f"\tSet paramaters to trainable")
        
        # Switch to GPU if available
        train_batch = batch_to_device(train_batch, device)

        if (cur_iter < 3): #Print out the first few iterations
            logger.log(f"\tTried to switch to GPU")

        # Zero the parameter gradients
        optimizer.zero_grad()

        if (cur_iter < 3): #Print out the first few iterations
            logger.log(f"\tZero_grad")

        # Forward propagation
        label_preds = model(train_batch['spectrum'], norm_in=True, denorm_out=False)
        
        if (cur_iter < 3): #Print out the first few iterations
            logger.log(f"\tForward propagation")
        
        # Compute mean-squared-error loss between predictions and normalized targets
        loss = torch.nn.MSELoss()(label_preds, model.normalize(train_batch['labels'], model.labels_mean, model.labels_std))
        
        if (cur_iter < 3): #Print out the first few iterations
            logger.log(f"\tCalc loss")

        # Back-propagation
        loss.backward()

        if (cur_iter < 3): #Print out the first few iterations
            logger.log(f"\tBack prop")
        
        # Weight updates
        optimizer.step()

        if (cur_iter < 3): #Print out the first few iterations
            logger.log(f"\tWeights update")
        
        # Save losses to find average later
        running_loss.append(float(loss))

        if (cur_iter < 3): #Print out the first few iterations
            logger.log(f"\tSave losses")
        
        cur_iter += 1
        
        # Display progress
        if cur_iter % verbose_iters == 0:
            logger.log('[Iter %i, %0.0f%%]' % (cur_iter, cur_iter/total_batch_iters*100))
            val_start_time = time.time()
            # Take average of training losses
            train_loss = np.nanmean(running_loss)
            losses['train_loss'].append(train_loss)
            logger.log('\tTrain Loss: %0.4f' % (train_loss))
            logger.log('\tTrain time taken: %0.0f seconds' % (val_start_time - iters_start_time))

            # Set parameters to not trainable
            model.eval()
            
            # Evaluate on validation set and display losses
            with torch.no_grad():
                for dataset in datasets:
                    running_loss = []

                    for val_batch in val_dataloaders[dataset]:
                        # Add noise to val_batch
                        if (ADD_NOISE == True):
                            val_batch['spectrum'] += torch.randn_like(val_batch['spectrum']) * std + mean

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

plotter = StarPlotter(output_dir, label_keys, datasets, saving=True)

plotter.plot_train_progress(cur_iter, losses)
logger.log("Plotted and saved progress")

# Load model info
if True == False:
    model_dir = ""
    load_model = os.path.join(model_dir,'cnn_synth_clean_1.pth.tar')

    checkpoint = torch.load(load_model, map_location=lambda storage, loc: storage)
    losses = dict(checkpoint['losses'])
    cur_iter = checkpoint['batch_iters']

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Load model weights
    model.load_state_dict(checkpoint['model'])


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
            label_preds = model(val_batch['spectrum'], 
                                norm_in=True, 
                                denorm_out=True)

            # Save batch data for comparisons
            ground_truth_labels[dataset].append(val_batch['labels'].cpu().data.numpy())
            model_pred_labels[dataset].append(label_preds.cpu().data.numpy())
        
        ground_truth_labels[dataset] = np.concatenate(ground_truth_labels[dataset])
        model_pred_labels[dataset] = np.concatenate(model_pred_labels[dataset])
        logger.log("\tPredicted labels for " + dataset)




plotter.plot_losses(model_pred_labels, ground_truth_labels)
logger.log("Plotted and saved losses")
plotter.plot_isochrones(model_pred_labels)
logger.log("Plotted and saved isochrones")
logger.log("Done!")

logger.close()



