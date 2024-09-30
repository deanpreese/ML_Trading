import numpy as np
import pywt
import matplotlib.pyplot as plt
import pandas as pd



datafile = [ 
        'data/Lucky13_3070_oos.csv',   
        'data/Lucky13_3070.csv',  #1
        'data/Lucky13_EX_3070_oos.csv',  
        'data/Lucky13_EX_3070.csv',  #3
        'data/Ind_F.csv',  #4
    ]

file_path = datafile[4]

df = pd.read_csv(file_path)
y = df['output'].values

# perform CWT
wavelet = "cmor1.5-1.0"
# logarithmic scale for scales, as suggested by Torrence & Compo:
widths = np.geomspace(1, 1024, num=100)
sampling_period = 1000
cwtmatr, freqs = pywt.cwt(y, widths, wavelet, sampling_period=sampling_period)
# absolute take absolute value of complex result
cwtmatr = np.abs(cwtmatr[:-1, :-1])


# plot result using matplotlib's pcolormesh (image with annoted axes)
fig, axs = plt.subplots(2, 1)
pcm = axs[0].pcolormesh(1000, freqs, cwtmatr)
axs[0].set_yscale("log")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Frequency (Hz)")
axs[0].set_title("Continuous Wavelet Transform (Scaleogram)")
fig.colorbar(pcm, ax=axs[0])

# plot fourier transform for comparison
from numpy.fft import rfft, rfftfreq

yf = rfft(y)
xf = rfftfreq(len(y), sampling_period)
plt.semilogx(xf, np.abs(yf))
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_title("Fourier Transform")
plt.tight_layout()

plt.show()

exit()



# Define the wavelet (we'll use a complex Morlet wavelet, 'cmor')
wavelet = 'mexh'

# Define the scale range (1 to 1000 for capturing various frequencies)
scales = np.arange(1, 128)

# Sampling frequency (assumed)
sampling_frequency = 100  # Example: 1000 Hz

# Apply Continuous Wavelet Transform (CWT)
coefficients, frequencies = pywt.cwt(y, scales, wavelet, sampling_period=1.0/sampling_frequency)

# Calculate center frequency of the wavelet
# 'cmor' has an inherent center frequency defined in PyWavelets
center_freq = pywt.central_frequency(wavelet)
print(f"Center Frequency of the Wavelet: {center_freq} Hz")

# Convert the scales to frequencies using PyWavelets' scale2frequency
wavelet_frequencies = pywt.scale2frequency(wavelet, scales) * sampling_frequency
print(f"Wavelet Frequencies at Various Scales (in Hz): {wavelet_frequencies}")

# Calculate the bandwidth (approximate) for each scale
# The bandwidth is related to the wavelet's ability to capture a range of frequencies.
# For simplicity, we assume the bandwidth is centered around the center frequency
bandwidths = wavelet_frequencies / scales  # This is a rough estimate
print(f"Approximate Bandwidths at Various Scales: {bandwidths}")

# Plotting the frequencies across scales
plt.figure(figsize=(10, 6))
plt.plot(scales, wavelet_frequencies, label='Center Frequency')
plt.fill_between(scales, wavelet_frequencies - bandwidths/2, wavelet_frequencies + bandwidths/2, 
                 color='gray', alpha=0.5, label='Bandwidth Range')
plt.title('Wavelet Center Frequency and Bandwidth at Different Scales')
plt.xlabel('Scale')
plt.ylabel('Frequency (Hz)')
plt.legend()
plt.show()
