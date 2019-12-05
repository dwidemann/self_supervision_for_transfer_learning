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
import pickle
#%%
def plot_synthetic_fft(fn,i,j):
    with open(fn, 'rb') as fid:
        D = pickle.load(fid)
        sig = np.squeeze(D[:,i,j])
    fs = 8116.0
    s = fn.split('_')
    num_sources = int(s[1])
    nfft = len(sig)
    #outfile = '{}_synthetic_FFT.png'.format(pngFn)
    S = rfft(sig, nfft)
    freqs = np.linspace(0,fs/2,len(S))
    #fig = figure(figsize=(10, 10))
    plt.plot(freqs,np.abs(S))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.title('Number of Sources{:d}, Mic Location Pixel {:d},{:d}'.format(num_sources,i,j))
    plt.show()
        #plt.savefig(outfile, dpi=300)


#%%
if __name__ == '__main__':
    fn = sys.argv[1]
    i = 50
    j = 30
    if len(sys.argv) > 2:   
        i,j = list(map(int,sys.argv[2].split(',')))
    plot_synthetic_fft(fn,i,j)
    
    
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

