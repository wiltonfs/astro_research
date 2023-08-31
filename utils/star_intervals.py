# # Starnet, with prediction intervals
# Felix Wilton
# 8/30/2023

import torch
from mapie.regression import MapieRegressor

from utils.star_model import *
from utils.star_MAPIE_wrapper import *

class StarNetConformalIntervals():
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = None
        
    def fit(self, star_dataset, iters=100, batch_size=16, initial_learning_rate=0.006, final_learning_rate=0.0005):
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
                train_batch = batch_to_device(train_batch, self.device)
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
                if cur_iter >= total_batch_iters:
                    break

        # Finished training
        print("Finished training!")
        model.eval()

        self.model = model

        print("MAPIE Time!")

        # MAPIE only handles single label regression:
        # We will already have a model trained
        # Make mapie_regressor for each label
        # Pass the train data in and "train" the regressor
        # Pass the test data in and "predict" the regressor
        # Compile the results into one output
        X_train = star_dataset.__toX__()
        self.mapie_regressors = []
        for label in star_dataset.label_keys:
            print("Training MAPIE regressor for label: ", label)
            mapie_regressor = MapieRegressor(StarNetSciKitMAPIE())
            y_train = star_dataset.__toY__(label)
            mapie_regressor.fit(X_train, y_train)
            self.mapie_regressors.append(mapie_regressor)
        
        
    def predict(self, star_dataset, alpha=0.1):
        with torch.no_grad():
            model_pred_labels = []
            dataloader = torch.utils.data.DataLoader(star_dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)
            for batch in dataloader:
                # Switch to GPU if available
                batch = batch_to_device(batch, self.device)
                # Forward propagation (and denormalize outputs)
                label_preds = self.model(batch['spectrum'], norm_in=True, denorm_out=True)
                # Save results
                model_pred_labels.append(label_preds.cpu().data.numpy())

        # Flatten the predictions into an n x 4 array
        model_pred_labels = np.concatenate(model_pred_labels, axis=0)

        print("MAPIE Time!")
        model_pred_intervals = np.zeros((len(model_pred_labels), len(star_dataset.label_keys), 2))
        X_pred = star_dataset.__toX__()
        for i, mapie_regressor in enumerate(self.mapie_regressors):
            print("Predicting MAPIE regressor for label: ", star_dataset.label_keys[i])
            Y_pred = model_pred_labels[:, i]
            prediction, prediction_interval = mapie_regressor.predict(X_pred, Y_pred, alpha=alpha)
            model_pred_intervals[:, i, 0] = prediction.squeeze() - prediction_interval[:, 0].squeeze()  # Calculate the lower errors
            model_pred_intervals[:, i, 1] = prediction_interval[:, 1].squeeze() - prediction.squeeze()  # Calculate the upper errors
        
        return model_pred_labels, model_pred_intervals

