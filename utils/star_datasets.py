# # Datasets and dataset utility functions
# Felix Wilton
# 7/22/2023
import torch
import h5py
import numpy as np

class SimpleSpectraDataset(torch.utils.data.Dataset):
    """
    Dataset loader for the simple spectral datasets.
    """

    def __init__(self, data_file, dataset, label_keys = ['teff', 'feh', 'logg', 'alpha'], noise_mean=0., noise_std=0.):
        self.data_file = data_file
        self.dataset = dataset.lower()
        self.label_keys = label_keys
        self.num_labels = len(label_keys)
        # Determine the number of pixels in each spectrum
        # Collect mean and std of the training data for normalization
        with h5py.File(data_file, "r") as f:
            self.num_spectra = len(f['spectra %s' % self.dataset])
            self.num_pixels = f['spectra %s' % self.dataset].shape[1]
            self.labels_mean = [np.nanmean(f[k + ' train'][:]) for k in label_keys]
            self.labels_std = [np.nanstd(f[k + ' train'][:]) for k in label_keys]
            self.spectra_mean = np.nanmean(f['spectra train'][:]) + noise_mean
            self.spectra_std = np.nanstd(f['spectra train'][:]) + noise_std
                        
    def __len__(self):
        return self.num_spectra
    
    def __getitem__(self, idx):
        
        with h5py.File(self.data_file, "r") as f: 
                
            # Load spectrum
            spectrum = f['spectra %s' % self.dataset][idx]
            spectrum[spectrum<-1] = -1.
            spectrum = torch.from_numpy(spectrum.astype(np.float32))
            
            # Load target stellar labels
            data_keys = f.keys()
            labels = []
            for k in self.label_keys:
                data_key = k + ' %s' % self.dataset
                if data_key in data_keys:
                    labels.append(f[data_key][idx])
                else:
                    labels.append(np.nan)
            labels = torch.from_numpy(np.asarray(labels).astype(np.float32))
            
        # Return full spectrum and target labels
        return {'spectrum':spectrum,
                'labels':labels}
    

    
    def __toX__(self):
        '''Return all spectra as a numpy array.'''
        with h5py.File(self.data_file, "r") as f:
            return f['spectra %s' % self.dataset][:]
        
    def __toY__(self, label_key):
        '''Return all labels as a numpy array.'''
        with h5py.File(self.data_file, "r") as f:
            return f[label_key + ' %s' % self.dataset][:]
        
    


