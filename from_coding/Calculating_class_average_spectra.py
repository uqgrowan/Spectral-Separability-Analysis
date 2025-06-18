import numpy as np
import pandas as pd

data = pd.read_csv(r"data\all_targets_normalized550_formatted.csv", index_col= [0])

classes = data["Class"].unique()

averages = {}
for clas in classes:        #clas instead of class because of weird built in?
    measurements = data[data["Class"] == clas].iloc[:, 1:502]
    averages[clas ] = np.mean(measurements, axis = 0)

class_averages = pd.DataFrame.from_dict(averages)
class_averages.to_csv("all_targets_all_classes_averaged.csv")