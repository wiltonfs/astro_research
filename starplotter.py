# # Plot model outputs, including training progress, performance on validation sets, and isochrones
# Felix Wilton
# 7/22/2023
import matplotlib.pyplot as plt

class StarPlotter():
    
    def __init__(self, saving_dir, saving=True):
        self.SAVING = saving
        self.dir = saving_dir



    
    # Plot train and validation loss over the iterations
    def plot_train_progress(self):
        plt.figure(figsize=(8,3))

        # Determine iteration of checkpoint
        batch_iters = np.linspace(cur_iter/len(losses['train_loss']), 
                                cur_iter, len(losses['train_loss']))

        # Plot training and validation progress
        plt.plot(batch_iters, losses['train_loss'], label='Training')
        plt.plot(batch_iters, losses['val_loss'], label='Validation')
        plt.xlim(batch_iters[0], batch_iters[-1])
        plt.ylim(0, 1.1*losses['val_loss'][2])
        plt.legend(fontsize=12)
        plt.grid()
        plt.title("Training Progress")
        plt.show()

        if self.SAVING:
            plt.savefig(self.dir + "TrainProgress.png")

        