# # Plot and Analyze Grid Search Results
# Felix Wilton
# 7/31/2023

import csv
import matplotlib.pyplot as plt

grid_ID = 'EpochNorm'
outPath = 'remoteOutputs/'
variables = ["batch size","iters","epochs", "initial learning rate", "final learning rate"]



data = []
with open(outPath + 'results.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    for row in csvreader:
        data.append(row)

val_loss_ID = header.index("val_loss") #search header for 'val_loss' and put that index as the ID

# Remove any values without grid_ID somewhere in their project_name
data = [row for row in data if grid_ID in row[0]]

# Sort data by val_loss (largest to smallest)
sorted_data = sorted(data, key=lambda row: float(row[val_loss_ID]), reverse=True)

# Plot the data without connecting the markers
plt.figure(figsize=(10, 6))  # Adjust figure size if needed
x_positions = range(len(sorted_data))  # x-axis positions spaced from left to right
val_loss = [float(row[val_loss_ID]) for row in sorted_data]
val_loss_std = [float(row[val_loss_ID+1]) for row in sorted_data]

plt.errorbar(x_positions, val_loss, yerr=val_loss_std, marker='o', linestyle='none')  # Remove the line connecting markers

plt.xlabel('Trials')
plt.ylabel('Validation Loss')
plt.title(f'Validation Loss for Each Trial in {grid_ID}')
plt.xticks(range(len(sorted_data)), [row[0] for row in sorted_data], rotation=45, ha='right')  # Set project names on x-axis
plt.tight_layout()  # To prevent label clipping
plt.savefig(f'{outPath}{grid_ID}/losses.png')

# Plot each variable in variables vs val_loss
for variable in variables:
    variable_ID = header.index(variable)
    plt.figure(figsize=(10, 6))
    x = [float(row[variable_ID]) for row in sorted_data]  # Extract and convert variable values
    y = [float(row[val_loss_ID]) for row in sorted_data]  # Extract and convert validation loss values
    plt.scatter(x, y)
    plt.xlabel(variable)
    plt.ylabel('Validation Loss')
    plt.title(f'Validation Loss vs {variable}')
    plt.tight_layout()  # To prevent label clipping
    plt.savefig(f'{outPath}{grid_ID}/{variable}.png')

# Print the projects from grid_ID with the 3 lowest val_loss values
print(f"Trials from {grid_ID} with the 3 lowest val_loss:")
for i in range(3):
    i = -1*(i+1)
    project_name = sorted_data[i][0]
    val_loss = float(sorted_data[i][val_loss_ID])
    print(f"{project_name}: {val_loss:.5f}")

a = 0
while True:
    a += 1
