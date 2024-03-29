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
from glob import glob
import pickle
import multiprocessing as mp 
from time import time, sleep
from scipy.signal import blackman
from datetime import datetime, timedelta
#%%

def save_fft(fn= 'Phoenix_Mar13-134312.bin',channel=5,outdir='acoustic_ffts'):
    numChannels = 8 # audio channels are 4-7 (vibrational are 0-3)
    fs = 51200
    f0 = 1002.05
    harmonics = [1,2,4]
    bandwidth = .5
    pngFn = os.path.basename(fn).replace('.bin','')
    s = pngFn.replace('Phoenix_','')
    outfiles = []
    outfiles.append('{}_channel_{:d}_fft.pkl'.format(s,channel))
    data = np.fromfile(fn, dtype='float32')
    D = data.reshape(len(data)//numChannels,numChannels)
    T = D.shape[0]
    halfway_seconds = int(np.round((T//2)/fs))
    if 'Mar' in s:
        month = 3
    else:
        month = 4
    day = int(s[3:5])
    hour = int(s[6:8])
    minute = int(s[8:10])
    second =  int(s[10:12])
    dt = datetime(year=2018,month=month,day=day,hour=hour,minute=minute,second=second)
    dt_plus10 = dt + timedelta(seconds=halfway_seconds)
    snew = 'Mar'
    if dt_plus10.month == 4:
        snew = 'Apr'
    snew = snew + dt_plus10.strftime("%d") + '_' + dt_plus10.strftime("%H") + \
           dt_plus10.strftime("%M") + dt_plus10.strftime("%S")
    outfiles.append('{}_channel_{:d}_fft.pkl'.format(snew,channel))
    for idx,outfile in enumerate(outfiles):
        if idx == 0:
            sig = D[:T//2,channel]
        else:
            sig = D[T//2:,channel]
        blackman_window = blackman(len(sig))
        N = len(sig)
        nfft = int(2**np.ceil(np.log2(N)))
        S = rfft(blackman_window*sig, nfft)
        freqs = np.linspace(0,fs/2,len(S))
        kept_freqs = []
        for idx in harmonics:
            kept_freqs.append((idx*f0-bandwidth/2,idx*f0+bandwidth/2))
        L = []
        for freqRange in kept_freqs:
            inds = np.where((freqs > freqRange[0]) & \
                            (freqs <freqRange[1]))[0]
            L.append((freqs[inds],S[inds]))        
        minL = np.infty
        for a,b in L:
            if len(a) < minL:
                minL = len(a)
        newL = []
        for a,b in L:
            newL.append((a[:minL],b[:minL]))

        f,sig = zip(*newL)
        f = np.array(f)
        sig = np.array(sig)
        with open(os.path.join(outdir,outfile),'wb') as fid:
            pickle.dump({'sig':sig,'freqs':f},fid)


def mp_worker(d):
    fn,channel,outdir = d
    t0 = time()
    try:
        save_fft(fn,channel,outdir)
        #sleep(channel)
        print("sample {} runtime: {:0.2f}".format(fn,time()-t0))
    except:
        print('error {}'.format(fn))


#%%
if __name__ == '__main__':
    debug = False # True
    def mp_handler():
        p = mp.Pool(num_cpus)
        p.map(mp_worker, queue)

    num_cpus = 30
    channel = 5
    direc = sys.argv[1]
    outdir = sys.argv[2]
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if os.path.isdir(direc):
        fns = glob(os.path.join(direc,'*','*.bin'))
    else:
        fns = [direc]
    queue = []
    for f in fns:
        if debug:
            channel = np.random.randint(1,6)
        queue.append((f,channel,outdir))

    mp_handler()
    
    
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

