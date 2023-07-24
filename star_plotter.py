# # Plot model outputs, including training progress, performance on validation sets, and isochrones
# Felix Wilton
# 7/22/2023
import matplotlib.pyplot as plt
import numpy as np
import os

### Plotting helper functions
def pretty(label):
    label_fancy = label
    
    if label=='teff':
        label_fancy = 'T$_{\mathrm{eff}}$ [K]'
    if label=='feh':
        label_fancy = '[Fe/H]'
    if label=='logg':
        label_fancy = 'log(g)'
    if label=='alpha':
        label_fancy = r'[$\alpha$/H]'
        
    if label=='synth_clean':
        label_fancy = 'Synthetic Data'
    if label=='synth_noised':
        label_fancy = 'Synthetic Data, Added Noise'
    if label=='obs_GAIA':
        label_fancy = 'Observed Data, GAIA Labels'
    if label=='obs_APOGEE':
        label_fancy = 'Observed Data, APOGEE Labels'
        
    return label_fancy

def getColor(dataset):
    color = 'blue'
    
    if dataset=='synth_clean':
        color = 'violet'
    if dataset=='synth_noised':
        color = 'violet'
    if dataset=='obs_GAIA':
        color = 'forestgreen'
    if dataset=='obs_APOGEE':
        color = 'forestgreen'
    
    return color


class StarPlotter():
    
    def __init__(self, saving_dir, label_keys, datasets, saving=True):
        self.dir = saving_dir
        self.label_keys = label_keys
        self.datasets = datasets
        self.SAVING = saving
    
    # Plot train and validation loss over the iterations
    def plot_train_progress(self, cur_iter, losses):
        plt.figure(figsize=(8,3))

        # Determine iteration of checkpoint
        batch_iters = np.linspace(cur_iter/len(losses['train_loss']), cur_iter, len(losses['train_loss']))

        # Plot training and validation progress
        plt.plot(batch_iters, losses['train_loss'], label='Training', color='black', linewidth=2)
        ylim = 0
        for dataset in self.datasets:
            plt.plot(batch_iters, losses['val_loss_'+dataset], color = getColor(dataset), label=pretty(dataset) + ' Validation')
            ylim = max(losses['val_loss_'+dataset][2], ylim)

        plt.xlim(batch_iters[0], batch_iters[-1])
        plt.ylim(0, 1.1*ylim)
        plt.legend(fontsize=12)
        plt.grid()
        plt.title("Training Progress")
        if self.SAVING:
            path = os.path.join(self.dir,"trainProgress.png")
            plt.savefig(path)
        plt.show()

    # Plot the isochrones
    def plot_isochrones(self, model_pred_labels):
        # Create the main figure and set the title
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('Isochrones', fontsize=16, fontweight='bold')

        # Iterate through the labels and create subplots
        for j, dataset in enumerate(self.datasets):
            # Create a subplot in the 2x2 grid
            ax = fig.add_subplot(2, 2, j+1)

            scatter = ax.scatter(model_pred_labels[dataset][:,0], model_pred_labels[dataset][:,2], c=model_pred_labels[dataset][:,1], cmap='viridis', s=0.4)
            
            # Customize each subplot
            pretty_dataset = pretty(dataset)
            ax.set_title(pretty_dataset)
            ax.set_xlabel(pretty('teff'), size=4*len(self.label_keys))
            ax.set_ylabel(pretty('logg'), size=4*len(self.label_keys))

            # Show colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(pretty('feh'), rotation=90, labelpad=15)
            
            ax.set_ylim(6, 0)
            ax.set_xlim(7000, 2500)

        # Adjust the spacing between subplots
        fig.tight_layout()
        

        # Save the figure
        if self.SAVING:
            path = os.path.join(self.dir,'isochrones.png')
            plt.savefig(path, facecolor='white', transparent=False, dpi=100, bbox_inches='tight', pad_inches=0.05)
        plt.show()

    #Plot performance on Validation Sets
    def plot_losses(self, model_pred_labels, ground_truth_labels):
        y_lims = [1000, 1.2, 1.5, 0.8]
        x_lims = [[2000, 9000],
                [-5.1, 1.1],
                [-1, 6],
                [-0.5, 0.9]]

        for i, label in enumerate(self.label_keys):
            # Create the main figure and set the title
            pretty_label = pretty(label)
            fig = plt.figure(figsize=(15, 8))
            fig.suptitle(pretty_label, fontsize=16, fontweight='bold')

            # Iterate through the labels and create subplots
            for j, dataset in enumerate(self.datasets):
                # Create a subplot in the 2x2 grid
                ax = fig.add_subplot(2, 2, j+1)
                
                # Calculate residual
                diff = model_pred_labels[dataset][:,i] - ground_truth_labels[dataset][:,i]
                
                # Create scatter plot on the given axes
                color = getColor(dataset)
                ax.scatter(ground_truth_labels[dataset][:,i], diff, alpha=0.5, s=5, zorder=1, c=color)

                # Customize each subplot
                pretty_dataset = pretty(dataset)
                ax.set_title(pretty_dataset)
                ax.set_xlabel(pretty_label, size=4*len(self.label_keys))
                ax.set_ylabel(r'$\Delta$ %s' % pretty_label, size=4*len(self.label_keys))
                
                # Add mean and spread information
                bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=1)
                if 'eff' in label:
                    ax.annotate('$\widetilde{m}$=%0.0f $s$=%0.0f'% (np.nanmean(diff), np.nanstd(diff)),
                                (0.75,0.8), size=4*len(self.label_keys), xycoords='axes fraction', 
                                bbox=bbox_props)
                else:
                    ax.annotate('$\widetilde{m}$=%0.2f $s$=%0.2f'% (np.nanmean(diff), np.nanstd(diff)),
                            (0.75,0.8), size=4*len(self.label_keys), xycoords='axes fraction', 
                            bbox=bbox_props)
                
                ax.axhline(0, linewidth=2, c='black', linestyle='--')
                ax.set_ylim(-y_lims[i], y_lims[i])
                ax.set_xlim(x_lims[i][0], x_lims[i][1])
                ax.set_yticks([-y_lims[i], -0.5*y_lims[i], 0, 0.5*y_lims[i], y_lims[i]])
                ax.tick_params(labelsize=2.8*len(self.label_keys))
                ax.grid()
            
            # Adjust the spacing between subplots
            fig.tight_layout()

            # Save the figure
            if self.SAVING:
                path = os.path.join(self.dir,label + '.png')
                plt.savefig(path, facecolor='white', transparent=False, dpi=100, bbox_inches='tight', pad_inches=0.05)
            plt.show()



            

