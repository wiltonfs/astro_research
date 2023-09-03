# astro_research
Stellar parameter predictions using supervised learning.

I built this repository between May and September of 2023 as an undergraduate researcher in Kwang Moo Yi's Computer Vision Lab at the University of British Columbia (UBC). My work was based on Teaghan O'Briain's previous work: github.com/teaghan/StarNet_SS.

## What is this for:
Take stellar emission spectra and train a simple, reliable neural network architecture to predict stellar parameters.

## How to use this:
### Main Python Scripts
* Use starnet.py to train a model:
  *  Command-line control over the majority of hyperparameters
  *  Detailed progress tracking/logging
  *  WandB integration
  *  Evaluation on other datasets
* Use starnet-analysis.py to generate figures for a single model run:
  *  Training progress on train, test, and evaluation sets
  *  Loss violin plots for synthetic data sets
  *  Loss scatter plots for all data sets
  *  Isochrones
* Use starnet-intervals.py to integrate a model with MAPIE:
  * Prediction intervals
  * Plotting with error bars
* Use hyperparam-search-analysis.py to generate figures for a hyperparameter search:
  * Identify best performing models
  * Plot loss of all models
  * Plot a selected combination of independent and dependant variables (hyperparameters)
### Job Scripts
These job scripts are specifically created for running on Compute Canada.
* Use starnet.sh to run starnet.py and stanet-analysis.py with command line hyperparameters on a GPU node
* Use hyperparam-search.sh to launch multiple sub-jobs (sub-search-script.sh) for hyperparameter searches (random search)
* Use starnet-intervals.sh to run starnet-intervals with a GPU node
### Utils
The utils folder has 5 helper scripts with useful functionality for datsets, logging, plotting, etc. It includes star_scikit.py, which wraps the model in the scikit API to enable functionality with MAPIE and other scikit-learn-compatible tools.

## Getting started:
1. Install this repo
2. Make sure you have numpy, torch, h5py, wandb, mapie
3. Make sure you have the 3 data files in a data directory
4. Try running starnet.py with the default settings
5. Try running starnet-analysis.py on the project starnet.py created