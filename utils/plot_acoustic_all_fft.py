#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:50:30 2019

@author: widemann1
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import rfft
from matplotlib.pyplot import figure
#%%

def plot_all_fft(fn):
    numChannels = 8 # audio channels are 4-7 (vibrational are 0-3)
    fs = 52100
    channels = [4,5,6,7]
    pngFn = os.path.basename(fn).replace('.bin','')
    s = pngFn.replace('Phoenix_','')
    outfile = '{}_FFT.png'.format(s)
    data = np.fromfile(fn, dtype='float32')
    D = data.reshape(numChannels,len(data)//numChannels)
    N = D.shape[1]
    nfft = int(2**np.ceil(np.log2(N)))
    S = []
    for channel in channels:
        sig = D[channel]
        S.append(np.abs(rfft(sig, nfft)))
    S = np.array(S)
    freqs = np.linspace(0,fs/2,S.shape[1])
    fig = figure(figsize=(10,10))
    plt.plot(freqs,S.T,'--')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.legend(['channel {:d}'.format(i) for i in channels])
    plt.title(s)
    #plt.show()
    plt.savefig(outfile, dpi=300)



#%%
if __name__ == '__main__':
    fn = sys.argv[1]
    plot_all_fft(fn)
    
    
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

