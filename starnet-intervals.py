import numpy as np

import matplotlib.pyplot as plt
from utils.star_datasets import *
from utils.star_intervals import *

train_data_file = "data/synth_clean.h5"
train_dataset = SimpleSpectraDataset(train_data_file, 'train')
test_dataset = SimpleSpectraDataset(train_data_file, 'val')



model = StarNetConformalIntervals()
print("Started training!")
model.fit(train_dataset)
print("Finished training!")
predictions = model.predict(test_dataset)
print("Finished predicting!")

print(predictions)




