# # Datasets and dataset utility functions
# Felix Wilton
# 7/22/2023
import torch
class SimpleSpectraDataset(torch.utils.data.Dataset):
    """
    Dataset loader for the simple spectral datasets.
    """

    def __init__(self, data_file, dataset, label_keys):
        
        self.data_file = data_file
        self.dataset = dataset.lower()
        self.label_keys = label_keys
        # Determine the number of pixels in each spectrum
        self.num_pixels = self.determine_num_pixels()
        self.num_spectra = self.determine_num_spectra()
                        
    def __len__(self):
        return self.num_spectra
    
    def determine_num_spectra(self):
        with h5py.File(self.data_file, "r") as f:    
            count = len(f['spectra %s' % self.dataset])
        return count
    
    def determine_num_pixels(self):
        with h5py.File(self.data_file, "r") as f:    
            pixels = f['spectra %s' % self.dataset].shape[1]
        return pixels
    
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
    
def batch_to_device(batch, device):
    '''Convert a batch of samples to the desired device.'''
    for k in batch.keys():
        if isinstance(batch[k], list):
            for i in range(len(batch[k])):
                batch[k][i] = batch[k][i].to(device)
        else:
            try:
                batch[k] = batch[k].to(device)
            except AttributeError:
                batch[k] = torch.tensor(batch[k]).to(device)
    return batch

