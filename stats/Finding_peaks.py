#Finding peaks

import pandas as pd
import numpy as np
from scipy.signal import find_peaks as fp
from scipy.signal import savgol_filter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import itertools

# Define data path
path = r"Combined_analysis\MEL_NZ_TAS_spectra.csv"

# Import data and drop unused column
df = pd.read_csv(path).drop(["setup"], axis = 1)

# Calculate mean spectra by class and site
means = df.groupby(["Class", "site"]).mean().transpose()

# Scale the spectra to account for illumination
scaler = RobustScaler().fit(means[75:600])
scaled = pd.DataFrame(scaler.transform(means))

# Relabel for plotting
scaled.index = df.iloc[:,2:].columns.astype("int64")
scaled.columns = means.columns

# Set bad bnds to NaN
scaled.loc[753:769, :] = np.nan

# Apply smoothing filter and prep for plotting 
smoothed = pd.DataFrame(savgol_filter(scaled, axis = 0, window_length= 5, polyorder= 1)) 
smoothed.index = df.iloc[:,2:].columns.astype("int64")
smoothed.columns = means.columns
smoothed.loc[753:769, : ] = np.nan
smoothed.loc[745:775, : ]

# Apply smoothing filter and prep for plotting
deriv_1 = pd.DataFrame(savgol_filter(scaled, axis = 0, window_length= 21, polyorder= 1, deriv = 1))
deriv_1.index = df.iloc[:,2:].columns.astype("int64")
deriv_1.columns = means.columns
deriv_1.loc[753:769, :] = np.nan

# Choose which dataset to plot
plotting_data = smoothed

# Plot all classes
for col in plotting_data.columns: 
    sns.lineplot(data = plotting_data, x = plotting_data.index, y = plotting_data.loc[:, col] )
#plt.xlim(left = 400, right = 900)
plt.ylim(bottom = -2 , top = 2)
plt.show()

# Calculate absolute value of derivative
absolute = np.absolute(deriv_1)

# Find peaks in peaks_data val of derivative
peaks_data = smoothed
col = 4
h = None
prom = 0.002
dist = 5
w = 3
rel_h = 0.5
peaks, properties = fp(x = peaks_data.iloc[:, col], 
                       height = h, 
                       prominence= prom,
                       distance = dist,
                       width = w,
                       rel_height = rel_h, 
                       )
len(peaks)

# Plot peaks_data and identified peaks
sns.lineplot(data = peaks_data, x = peaks_data.index, y = peaks_data.iloc[:, col] )
sns.scatterplot(data = peaks_data.iloc[peaks], x = peaks_data.iloc[peaks].index, y = peaks_data.iloc[peaks, col], color = "red")
plt.show()

# Make overall list of peaks for all class means
peaks_list = []
for col in peaks_data.columns : 
    peaks, _ = fp(x = peaks_data.loc[400:925, col], 
                       height = h, 
                       prominence= prom,
                       distance = dist,
                       width = w,
                       rel_height = rel_h
                       )
    peaks_list.append(list(peaks))

# Flatten list of lists of peaks   
peaks_list = list(itertools.chain.from_iterable(peaks_list))

# Summarize peaks list and reformat
values, counts = np.unique(peaks_list, return_counts= True)
peak_counts = pd.DataFrame([values, counts]).transpose().set_index(0)
peak_counts.columns = ["counts"]
peak_counts.index = peak_counts.index +400

# Calculate peak count over wavelength window
width_range = 4 #Must be even
sums = []
for row in range(0, peak_counts.shape[0]):
    n = peak_counts.index[row]
    bands = [num for num in range (int(n - width_range/2), int(n + width_range/2)+1)]
    band_sum = np.sum(peak_counts[peak_counts.index.isin(bands)].iloc[:, 0])
    sums.append(band_sum)

# Add window sums to dataframe   
peak_counts["neighbourhood"] = sums

# Export peaks data
peak_counts.to_csv("smoothed_peaks_counts.csv")


#Plot peak_counts graph
colour_dict_purple_yellow = {
    1: "#292f56",
    2: "#413665",
    3: "#5b3b71",
    4: "#773f7a",
    5: "#94437d",
    6: "#af477d",
    7: "#c84e78",
    8: "#de586f",
    9: "#ef6764",
    10: "#fc7a56",
    11: "#ff9047",
    12: "#ffa836",
    13: "#fdc124", 
    14: "#f0db19",
    }
colour_dict_dgrey_red = {
    1: "#616161",
    2: "#6f5f63",
    3: "#7d5c63",
    4: "#8a5963",
    5: "#975561",
    6: "#a3515e",
    7: "#af4c59",
    8: "#ba4754",
    9: "#c5414e",
    10: "#cf3a46",
    11: "#d9333d",
    12: "#e12b33",
    13: "#e92228",
    14: "#f01919",
    }

colour_dict_lgrey_red= {
    1: "#ededed",
    2: "#f2dfe3",
    3: "#f7d1d9",
    4: "#fbc3cc",
    5: "#ffb5bf",
    6: "#ffa6b1",
    7: "#ff97a1",
    8: "#ff8890",
    9: "#ff797f",
    10: "#ff696d",
    11: "#ff595a",
    12: "#fb4746",
    13: "#f63331",
    14: "#f01919",
    }


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.spines["bottom"].set_position("center")
ax.spines[["right", "top", "left"]].set_visible(False)
for index in peak_counts.index:
    plt.vlines(x= index, ymin= -0.5,ymax = 0.5, colors= colour_dict_lgrey_red[peak_counts.loc[index, "counts"]])
plt.yticks(ticks= [], labels= [])
plt.show()
