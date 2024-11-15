import pywt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

datafile = [ 
    'data/Lucky13_3070_oos.csv',   
    'data/Lucky13_3070.csv',  #1
    'data/Lucky13_EX_3070_oos.csv',  
    'data/Lucky13_EX_3070.csv',  #3
    'data/Lucky13_ALL_oos.csv',  #4
    'data/Lucky13_ALL.csv',  #5
    'data/oos_o.csv',  #6
    
]

df = pd.read_csv(datafile[6])

wavelet = "cmor1.5-1.0"
#wavelet = "db6"

# Perform the Discrete Wavelet Transform with 'dbN'
wavelet = pywt.Wavelet('db6')
coeffs = pywt.wavedec(df['actual'], wavelet, level=7)

# Plotting the Wavelet Coefficients using imshow
fig, axes = plt.subplots(len(coeffs), 1, figsize=(12, 7))

for i, ax in enumerate(axes):
    im = ax.imshow(np.abs(coeffs[i].reshape(1, -1)), aspect='auto', cmap='jet')
    ax.set_title(f'Coefficient level {i}')

# Add a single colorbar on the far right
fig.subplots_adjust(right=0.85)
#cbar_ax = fig.add_axes([0.86, 0.15, 0.03, 0.7])
#fig.colorbar(im, cax=cbar_ax)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.show()