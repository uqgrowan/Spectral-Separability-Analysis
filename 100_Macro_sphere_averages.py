#Plotting macrocystis spectra averaged by blade number

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cm

data = pd.read_csv(r"C:\Users\s4770224\Documents\Work\Writing\Macro_avg_by_BladeNumber.csv")
wavelengths = data.iloc[:, 0]
spectra = data.iloc[:, 1:]

# Number of spectra
num_spectra = len(spectra.columns)

# Create the colormap object
cmap = cm.Greens

plt.figure()
# Plot each spectrum with a color from the magma colormap
for i, col in enumerate(spectra.columns):
    color = cmap(i / (num_spectra - 1))  # Normalize index to [0, 1] range
    plt.plot(wavelengths, spectra[col], label=col, color=color)

plt.ylabel("Relative reflectance")
plt.xlabel("Wavelength")
plt.ylim(0, 0.2)
plt.xlim(400, 1800)
plt.legend()
plt.show()