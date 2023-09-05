# # Plot model outputs, including training progress, performance on validation sets, and isochrones
# Felix Wilton
# 7/22/2023
import matplotlib.pyplot as plt
import numpy as np
import os

### Plotting helper functions
def pretty(label):
    '''
    Turns file-friendly label names into pretty label text
    '''
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
        label_fancy = 'Synthetic'
    if label=='synth_noised':
        label_fancy = 'Synthetic, Added Noise'
    if label=='obs_GAIA':
        label_fancy = 'Observed, GAIA Labels'
    if label=='obs_APOGEE':
        label_fancy = 'Observed, APOGEE Labels'
        
    return label_fancy

def getColor(dataset):
    '''
    Get graphing color for each dataset
    '''
    color = 'blue'
    
    if dataset=='synth_clean':
        color = 'violet'
    if dataset=='synth_noised':
        color = 'violet'
    if dataset=='obs_GAIA':
        color = 'forestgreen'
    if dataset=='obs_APOGEE':
        color = 'cornflowerblue'
    
    return color

def getLinestyle(dataset):
    '''
    Get linestyle for each dataset
    '''
    style = '-'
    
    if dataset=='synth_clean':
        style = '-'
    if dataset=='synth_noised':
        style = ':'
    if dataset=='obs_GAIA':
        style = '-'
    if dataset=='obs_APOGEE':
        style = ':'
    
    return style


class StarPlotter():
    '''
    Plot model outputs, including training progress, performance on validation sets, and isochrones
    '''
    
    def __init__(self, saving_dir, label_keys = ['teff', 'feh', 'logg', 'alpha'], datasets=None, saving=True):
        self.dir = saving_dir
        self.label_keys = label_keys
        self.datasets = datasets
        self.SAVING = saving
    
    def plot_train_progress(self, losses, std=False):
        '''
        Plot train and validation loss over the iterations
        '''
        plt.figure(figsize=(8,3))

        eval_alpha = 0.5
        std_alpha = 0.2

        # Plot training and validation progress
        plt.plot(losses['iter'], losses['train_loss'], label='Training', color='black', linewidth=4)
        plt.plot(losses['iter'], losses['val_loss'], label='Validation', color='red')

        if std:
            plt.fill_between(losses['iter'], losses['train_loss'] - losses['train_std'], losses['train_loss'] + losses['train_std'], color='black', alpha=std_alpha)
            plt.fill_between(losses['iter'], losses['val_loss'] - losses['val_std'], losses['val_loss'] + losses['val_std'], color='red', alpha=std_alpha)

        ylim = 0
        for dataset in self.datasets:
            plt.plot(losses['iter'], losses['eval_loss_' + dataset], color=getColor(dataset), label=pretty(dataset) + ' Evaluation', linestyle=getLinestyle(dataset), alpha=eval_alpha)
            #plt.fill_between(losses['iter'], losses['eval_loss_' + dataset] - losses['eval_std_' + dataset], losses['eval_loss_' + dataset] + losses['eval_std_' + dataset], color=getColor(dataset), alpha=std_alpha)
            ylim = max(losses['eval_loss_' + dataset][2], ylim)

        plt.xlim(losses['iter'][0], losses['iter'][-1])
        plt.ylim(0, 1.1 * ylim)
        plt.ylabel("Mean-squared-error loss")
        plt.xlabel("Iteration")
        # Move the legend to the right, outside the plot
        plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1.05, 1))
        plt.grid()
        plt.title("Training Progress")
        if self.SAVING:
            path = os.path.join(self.dir, "trainProgress.png")
            plt.savefig(path, facecolor='white', transparent=False, dpi=100, bbox_inches='tight', pad_inches=0.05)

    def plot_isochrones(self, model_pred_labels):
        '''
        Plot isochrones for each dataset
        '''
        # Create the main figure and set the title
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle('Isochrones', fontsize=16, fontweight='bold')

        # Initialize vmin and vmax for consistent color scale across subplots
        vmin = min(model_pred_labels[dataset][:, 1].min() for dataset in self.datasets)
        vmax = max(model_pred_labels[dataset][:, 1].max() for dataset in self.datasets)

        # Iterate through the labels and create subplots
        for j, dataset in enumerate(self.datasets):
            # Create a subplot in the 2x2 grid
            ax = fig.add_subplot(2, 2, j+1)

            scatter = ax.scatter(model_pred_labels[dataset][:, 0], model_pred_labels[dataset][:, 2], c=model_pred_labels[dataset][:, 1], cmap='viridis', s=0.4, vmin=vmin, vmax=vmax)

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
            path = os.path.join(self.dir, 'isochrones.png')
            plt.savefig(path, facecolor='white', transparent=False, dpi=100, bbox_inches='tight', pad_inches=0.05)

    def plot_violin_loss(self, model_pred_labels, ground_truth_labels):
        '''
        Plot performance on synthetic evaluation set as violin plot
        '''
        y_lims = [1000, 1, 1, 1, 10]

        fig, axes = plt.subplots(len(self.label_keys), 1, figsize=(12, len(self.label_keys)*2.7))
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=1)
        pred_stellar_labels = model_pred_labels['synth_clean']
        tgt_stellar_labels = ground_truth_labels['synth_clean']

        # Iterate through labels
        for i, ax in enumerate(axes):
            label_key = pretty(self.label_keys[i])
                
            # Calculate residual
            diff = pred_stellar_labels[:,i] - tgt_stellar_labels[:,i]
                
            # Determine boxplot info
            box_positions = []
            box_data = []
            for tgt_val in np.unique(tgt_stellar_labels[:,i]):
                indices = np.where(tgt_stellar_labels[:,i]==tgt_val)[0]
                if len(indices)>2:
                    box_positions.append(tgt_val)
                    box_data.append(diff[indices])
            box_width = np.mean(np.diff(box_positions))/2

            # Plot
            ax.violinplot(box_data, positions=box_positions, widths=box_width,
                        showextrema=True, showmeans=False)
            
            # Annotate median and standard deviation of residuals
            if 'eff' in label_key:
                ax.annotate('$\widetilde{m}$=%0.0f $s$=%0.0f'% (np.median(diff), np.std(diff)),
                            (0.75,0.8), size=4*len(self.label_keys), xycoords='axes fraction', 
                            bbox=bbox_props)
            elif 'rad' in label_key:
                ax.annotate('$\widetilde{m}$=%0.1f $s$=%0.1f'% (np.median(diff), np.std(diff)),
                            (0.75,0.8), size=4*len(self.label_keys), xycoords='axes fraction', 
                            bbox=bbox_props)
            else:
                ax.annotate('$\widetilde{m}$=%0.2f $s$=%0.2f'% (np.median(diff), np.std(diff)),
                        (0.75,0.8), size=4*len(self.label_keys), xycoords='axes fraction', 
                        bbox=bbox_props)
                
            # Axes parameters
            ax.set_xlabel('%s' % (label_key), size=4*len(self.label_keys))
            ax.set_ylabel(r'$\Delta$ %s' % label_key, size=4*len(self.label_keys))
            ax.axhline(0, linewidth=2, c='black', linestyle='--')
            ax.set_ylim(-y_lims[i], y_lims[i])
            ax.set_xlim(np.min(box_positions)-box_width*2, np.max(box_positions)+box_width*2)
            ax.set_yticks([-y_lims[i], -0.5*y_lims[i], 0, 0.5*y_lims[i], y_lims[i]])
            
            # Annotate sample size of each bin
            ax.text(box_positions[0]-2*box_width, 1.11*y_lims[i], 'n = ',
                fontsize=3*len(self.label_keys))
            ax_t = ax.secondary_xaxis('top')
            ax_t.set_xticks(box_positions)
            ax_t.set_xticklabels([len(d) for d in box_data])
            ax_t.tick_params(axis='x', direction='in', labelsize=3*len(self.label_keys))

            # Set Teff values to integers
            if 'eff' in label_key:
                tick_positions = ax.get_xticks()
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(np.array(tick_positions).astype(int))

            ax.tick_params(labelsize=3*len(self.label_keys))
            ax.grid()
        
        plt.tight_layout()
        plt.subplots_adjust(hspace=.5)

        # Save the figure
        if self.SAVING:
            path = os.path.join(self.dir,'violin.png')
            plt.savefig(path, facecolor='white', transparent=False, dpi=100, bbox_inches='tight', pad_inches=0.05)

    def plot_scatter_losses(self, model_pred_labels, ground_truth_labels):
        '''
        Plot performance on evaluation sets as a scatter plot
        '''
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


    def plot_scatter_losses_intervals(self, y_true, prediction, prediction_interval, label, alpha, iters, dataset="synth_clean"):
        '''
        Plot performance on evaluation set as a scatter plot with error bars
        '''
        y_lims = [1000, 1.2, 1.5, 0.8]
        x_lims = [[2000, 9000],
                [-5.1, 1.1],
                [-1, 6],
                [-0.5, 0.9]]
        label_keys = ['teff', 'feh', 'logg', 'alpha']
        i = label_keys.index(label)

        pretty_label = pretty(label)   
        yerr = np.ones((2, len(y_true)))
        yerr[0, :] = prediction.squeeze() - prediction_interval[:, 0].squeeze()  # Calculate the lower errors
        yerr[1, :] = prediction_interval[:, 1].squeeze() - prediction.squeeze()  # Calculate the upper errors
        yerr = np.abs(yerr)
        # Create a scatter plot with error bars
        plt.figure(figsize=(15, 8))
        color = getColor(dataset)
        plt.errorbar(y_true, prediction-y_true, yerr=yerr, fmt='s', alpha=0.5, markersize=4, label=label, color=color, ecolor=color)
        plt.xlabel(pretty_label)
        plt.ylabel(r'$\Delta$ %s' % pretty_label)
        plt.title(f"{pretty_label} (Error Bars Indicating {((1-alpha)*100):.1f}% Confidence)")
        plt.axhline(0, linewidth=2, c='black', linestyle='--')
        plt.ylim(-y_lims[i], y_lims[i])
        plt.xlim(x_lims[i][0], x_lims[i][1])
        plt.yticks([-y_lims[i], -0.5*y_lims[i], 0, 0.5*y_lims[i], y_lims[i]])
        plt.tick_params(labelsize=2.8*len(self.label_keys))
        plt.grid()
        plt.legend()
        
        if self.SAVING:
            path = self.dir + str(iters) + "_" + str(alpha) + "_" + dataset + "_" + label + "_results.png"
            plt.savefig(path, facecolor='white', transparent=False, dpi=100, bbox_inches='tight', pad_inches=0.05)

        plt.close()



            

