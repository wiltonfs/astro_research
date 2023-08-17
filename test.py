import matplotlib.pyplot as plt

# Create sample data
x = [1, 2, 3, 4, 5]
y = [10, 15, 7, 12, 9]

# Create a figure and save the figure to a specified path
plt.figure(figsize=(8,8))
plt.plot(x, y, label='Data')
plt.title('Larger Matplotlib Example')
plt.legend()
plt.savefig('outputs/PlottingTestImageLarge.png')

# Create a second figure and save to the same path
plt.figure(figsize=(2,2))
plt.plot(y, x, label='Data')
plt.title('Small Matplotlib Example')
plt.legend()
plt.savefig('outputs/PlottingTestImageSmall.png')