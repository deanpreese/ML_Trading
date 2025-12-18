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
    'data/Model_3LB_ALL_oos.csv' #7
]

df = pd.read_csv(datafile[7])

three_lb_filter_x = ["HourOfDay","MinOfHour",
                       "RSIRAW","SDLR310","SDBB91","SDKC91","SDKC9","ROC","ATR54","ATR53","ATR52",
                       "ATR51","ATR5","ATR21","ATR2","STOK1",       
                        'cv2', 'cv1', 'cv0', 'chv2', 'chv0', 'cv4', 'chv12', 'chv18', 'chv11']

wavelet = "cmor1.5-1.0"
#wavelet = "db6"

# Perform the Discrete Wavelet Transform with 'dbN'
wavelet = pywt.Wavelet('db6')
coeffs = pywt.wavedec(df['cv2'], wavelet, level=7)

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