class StarNet(nn.Module):
    def __init__(self, num_pixels, num_filters, filter_length, 
                 pool_length, num_hidden, num_labels,
                 spectra_mean, spectra_std, labels_mean, labels_std):
        super().__init__()
        
        # Save distribution of training data
        self.spectra_mean = spectra_mean
        self.spectra_std = spectra_std
        self.labels_mean = torch.tensor(np.asarray(labels_mean).astype(np.float32)).to(device)
        self.labels_std = torch.tensor(np.asarray(labels_std).astype(np.float32)).to(device)
        
        # Convolutional and pooling layers
        self.conv1 = nn.Conv1d(1, num_filters[0], filter_length)
        self.conv2 = nn.Conv1d(num_filters[0], num_filters[1], filter_length)
        self.pool = nn.MaxPool1d(pool_length, pool_length)
        
        # Determine shape after pooling
        pool_output_shape = compute_out_size((1,num_pixels), 
                                             nn.Sequential(self.conv1, 
                                                           self.conv2, 
                                                           self.pool))
        
        # Fully connected layers
        self.fc1 = nn.Linear(pool_output_shape[0]*pool_output_shape[1], num_hidden[0])
        self.fc2 = nn.Linear(num_hidden[0], num_hidden[1])
        self.output = nn.Linear(num_hidden[1], num_labels)

    def normalize(self, data, data_mean, data_std):
        '''Normalize inputs to have zero-mean and unit-variance.'''
        return (data - data_mean) / data_std
    
    def denormalize(self, data, data_mean, data_std):
        '''Undo the normalization to put the data back in the original scale.'''
        return data * data_std + data_mean
        
    def forward(self, x, norm_in=True, denorm_out=False):
        
        if norm_in:
            # Normalize spectra to have zero-mean and unit variance
            x = self.normalize(x, self.spectra_mean, self.spectra_std)
        
        x = F.relu(self.conv1(x.unsqueeze(1)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        
        if denorm_out:
            # Denormalize predictions to be on the original scale of the labels
            x = self.denormalize(x, self.labels_mean, self.labels_std)
            
        return x