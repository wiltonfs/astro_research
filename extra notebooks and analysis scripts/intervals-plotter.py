import matplotlib.pyplot as plt
import numpy as np

#TODO iterate through each of the files and calculate the actual confidence and log to a txt file
#TODO plot the actual confidence vs iters, and width vs iters (3rd axis is alpha)
#TODO try it on the observed data

## Plots I want to make:
# 1. Actual confidence vs iters (3rd axis is alpha, one for each dataset)
# 2. Width vs iters (3rd axis is alpha, one for each dataset)
test_data_files = ["synth_clean", "obs_APOGEE", "obs_GAIA"]
label_keys = ['teff', 'feh', 'logg', 'alpha']
alphas = [0.1, 0.5, 0.9]

resultsPath = "remoteOutputs/intervalOutputs2/intervalOutputs/"
outputPath = "intervalFigures/"

# Open overall_results.csv
with open(resultsPath + 'overall_results.csv', 'r') as file:
    lines = file.readlines()

# Extract and process data for the current dataset
header = lines[0].strip().split(',')
data = [line.strip().split(',') for line in lines[1:]]
# Find the indices of relevant columns in the header
dataset_idx = header.index('test_data_file')
iters_idx = header.index('iters')
label_idx = header.index('label')
alpha_idx = header.index('alpha')
actual_conf_idx = header.index('actual_conf')
avg_interval_width_idx = header.index('avg_interval_width')

# Plotting mark shapes
shapes = ["o", "s", "^"]

for dataset in test_data_files:
    # Only for test_data_file == datset
    # Plot actual_conf vs iters, with alpha as the color
    # Plot avg_interval_width vs iters, with alpha as the color
    # Save the plots
    
    # Initialize lists to store data points
    iters_data = []
    actual_conf_data = []
    avg_interval_width_data = []
    alpha_data = []
    
    for row in data:
        if row[dataset_idx] == dataset:
            iters_data.append(int(row[iters_idx]))
            iters_data_log = np.log10(iters_data)
            actual_conf_data.append(float(row[actual_conf_idx]))
            avg_interval_width_data.append(float(row[avg_interval_width_idx]))
            alpha_data.append(float(row[alpha_idx]))
    
    # Create a new figure for actual_conf vs iters
    plt.figure(figsize=(10, 6))
    for alpha in alphas:
        iters_data_alpha = []
        actual_conf_data_alpha = []
        for i in range(len(iters_data)):
            if alpha_data[i] == alpha:
                iters_data_alpha.append(iters_data_log[i])
                actual_conf_data_alpha.append(actual_conf_data[i])
        plt.scatter(iters_data_alpha, actual_conf_data_alpha, label=f'Confidence = {((1-alpha) * 100):.0f}%', alpha=0.5, marker=shapes[alphas.index(alpha)])
    plt.xlabel('log10 Iterations')
    plt.ylabel('Actual Confidence')
    plt.title(f'Actual Confidence vs Iterations for {dataset}')
    plt.xlim(1, 6)
    plt.ylim(0, 120)
    plt.grid(False)
    plt.legend()
    plt.savefig(f'{outputPath}actual_conf_vs_iters_{dataset}.png')
    plt.close()
    
    # Create a new figure for avg_interval_width vs iters
    plt.figure(figsize=(10, 6))
    for alpha in alphas:
        iters_data_alpha = []
        avg_interval_width_data_alpha = []
        for i in range(len(iters_data)):
            if alpha_data[i] == alpha:
                iters_data_alpha.append(iters_data_log[i])
                avg_interval_width_data_alpha.append(avg_interval_width_data[i])
        plt.scatter(iters_data_alpha, avg_interval_width_data_alpha, label=f'Confidence = {((1-alpha) * 100):.0f}%', alpha=0.5, marker=shapes[alphas.index(alpha)])
    plt.xlabel('log10 Iterations')
    plt.ylabel('Average Interval Width')
    plt.title(f'Average Interval Width vs Iterations for {dataset}')
    plt.xlim(1, 6)
    plt.ylim(-5, 5)
    plt.grid(False)
    plt.legend()
    plt.savefig(f'{outputPath}avg_interval_width_vs_iters_{dataset}.png')
    plt.close()

    





