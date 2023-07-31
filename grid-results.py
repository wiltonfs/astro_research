# # Plot and Analyze Grid Search Results
# Felix Wilton
# 7/31/2023

import csv
import matplotlib.pyplot as plt

grid_ID = 'Grid'

data = []
with open('outputs/results.csv', 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    for row in csvreader:
        data.append(row)

# Step 1: Remove any values without grid_ID somewhere in their project_name
data = [row for row in data if grid_ID in row[0]]

# Step 2: Sort data by val_loss (largest to smallest)
sorted_data = sorted(data, key=lambda row: float(row[7]), reverse=True)

# Plot the data without connecting the markers
plt.figure(figsize=(10, 6))  # Adjust figure size if needed
x_positions = range(len(sorted_data))  # x-axis positions spaced from left to right
val_loss = [float(row[7]) for row in sorted_data]
val_loss_std = [float(row[8]) for row in sorted_data]

plt.errorbar(x_positions, val_loss, yerr=val_loss_std, marker='o', linestyle='none')  # Remove the line connecting markers

plt.xlabel('Projects')
plt.ylabel('Validation Loss')
plt.title(f'Validation Loss for Each Project in {grid_ID}')
plt.xticks(range(len(sorted_data)), [row[0] for row in sorted_data], rotation=45, ha='right')  # Set project names on x-axis
plt.tight_layout()  # To prevent label clipping
plt.savefig(f'outputs/{grid_ID}_loss.png')
plt.show()

# Print the projects from grid_ID with the 3 lowest val_loss values
print(f"Projects from {grid_ID} with the 3 lowest val_loss:")
for i in range(3):
    i = -1*(i+1)
    project_name = sorted_data[i][0]
    val_loss = float(sorted_data[i][7])
    print(f"{project_name}: {val_loss:.5f}")
