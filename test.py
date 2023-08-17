import matplotlib.pyplot as plt

# Create sample data
x = [1, 2, 3, 4, 5]
y = [10, 15, 7, 12, 9]

# Create a figure and axis
fig, ax = plt.subplots()
ax.plot(x, y, label='Data')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Simple Matplotlib Example')
ax.legend()

# Save the figure to a specified path
save_path = 'outputs/PlottingTestImage.png'
plt.savefig(save_path)