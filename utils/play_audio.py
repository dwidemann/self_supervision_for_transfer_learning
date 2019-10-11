#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:50:30 2019

@author: widemann1
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
#%%
def play_audio(fn,channel):
    numChannels = 8 # audio channels are 4-7 (vibrational are 0-3)
    fs = 52100
    #fn = '/Users/widemann1/Desktop/test_audio/Phoenix_Mar27-113358.bin'
    #channel = 4
    pngFn = os.path.basename(fn).replace('.bin','')
    data = np.fromfile(fn, dtype='float32')
    D = data.reshape(len(data)//numChannels,numChannels)
    sig = D[:,channel]
    sd.play(sig,fs)


#%%
if __name__ == '__main__':
    fn = sys.argv[1]
    channel = int(sys.argv[2])
    play_audio(fn,channel)
    
    
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

