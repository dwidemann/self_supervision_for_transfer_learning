#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:50:30 2019

@author: widemann1
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

#%%
def plot_spectrogram(fn,channels):
    numChannels = 8 # audio channels are 4-7 (vibrational are 0-3)
    fs = 52100
    nfft = int(2**np.ceil(np.log2(fs)))
    #fn = '/Users/widemann1/Desktop/test_audio/Phoenix_Mar27-113358.bin'
    #channel = 4
    pngFn = os.path.basename(fn).replace('.bin','')
    data = np.fromfile(fn, dtype='float32')
    D = data.reshape(numChannels,len(data)//numChannels)
    if type(channels) == int:
        channels = [channels]
    for channel in channels:
        sig = D[channel,:20*fs]
        outfile = '{}_channel{:d}.png'.format(pngFn,channel)
        f, t, Sxx = spectrogram(sig, fs, nfft=nfft)
        plt.pcolormesh(t, f, Sxx)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
        #plt.savefig(outfile, dpi=300)


#%%
if __name__ == '__main__':
    fn = sys.argv[1]
    channels = list(map(int,sys.argv[2].split(',')))
    plot_spectrogram(fn,channels)
    
    
#%%

# from scipy.io import loadmat
# A = loadmat(matfile)
# matData = A['IoTechData']


# matfile = '/Users/widemann1/Documents/adapd/multimodal_feature_combinations/data_readers/acoustic_Mar27-113358.mat'

# fs = 52100
# time_samples = [0,12]

# inds = np.arange(fs*time_samples[0],int(fs*time_samples[1]))

# if 0:
#     plt.plot(matData[channel,inds])
#     plt.show()

#%%

