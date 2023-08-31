# # Scikit Wrapper for starnet
# Felix Wilton
# 8/30/2023

import torch
from sklearn.base import BaseEstimator

from utils.star_model import *

class StarNetSciKit(BaseEstimator):
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = None
        
    def fit(self, X, y, iters=1000, batch_size=16, initial_learning_rate=0.006, final_learning_rate=0.0005):
        # Train your StarNet model on star_dataset for iters iterations
        star_dataset = X
        total_batch_iters = iters
        cur_iter = 0

        train_dataloader = torch.utils.data.DataLoader(star_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        model = StarNet(self.device, star_dataset)
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), initial_learning_rate, weight_decay=0)

        # Calculate the learning rate schedule
        lr_factor = (final_learning_rate / initial_learning_rate) ** (1.0 / total_batch_iters)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_factor ** epoch)


        # Continuously loop over the training set
        while cur_iter < (total_batch_iters):
            for train_batch in train_dataloader:
                # Set parameters to trainable
                model.train()
                # Switch to GPU if available
                train_batch = star_dataset.batch_to_device(train_batch, self.device)
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
                cur_iter += 1

        # Finished training
        model.eval()
        
    def predict(self, X):
        star_dataset = X
        with torch.no_grad():
            model_pred_labels = []
            dataloader = torch.utils.data.DataLoader(star_dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)
            for batch in dataloader:
                # Switch to GPU if available
                batch = star_dataset.batch_to_device(batch, self.device)
                # Forward propagation (and denormalize outputs)
                label_preds = self.model(batch['spectrum'], norm_in=True, denorm_out=True)
                # Save results
                model_pred_labels.append(label_preds.cpu().data.numpy())