#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 09:05:04 2019

@author: widemann1
"""

import os, sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp2d

#%%
data_direc = '/Users/widemann1/Documents/adapd/autoencoder'
fn_ss = os.path.join(data_direc,'spectra_SS.pickle')
fn_aa = os.path.join(data_direc,'spectra_AA.pickle')
#%%
with open(fn_ss,'rb') as fid:
    spectra_ss = pickle.load(fid)

#%%
sig, label, c = spectra_ss

#%%


def plot_unlabeled_sample(data,outfile=None,title=None):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(data)
    ratio = 1.0
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    # the abs method is used to make sure that all numbers are positive
    # because x and y axis of an axes maybe inversed.
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    fig.colorbar(im)
    # or we can utilise the get_data_ratio method which is more concise
    # ax.set_aspect(1.0/ax.get_data_ratio()*ratio),
    if title:
        plt.title(title)
    if outfile:
        plt.savefig(fname=outfile,dpi=300,format='png')
    else:
        plt.show()
        
#%%
data = np.array(sig[-1])
plot_unlabeled_sample(data)

#%%
def transform_sig(data,interp=True):
        inds = np.arange(2,101,6)
        sig = data[inds,:]
        A = np.abs(sig)
        A = np.log10(A)
        
        mu = A.mean(axis=1).reshape(len(inds),1)
        std = A.std(axis=1).reshape(len(inds),1)
        
        data = (A-mu)/std
        if interp:
            num_freq_bins = 1342
            interp = interp2d(range(num_freq_bins),
                              inds,
                              data,kind='linear')
            new_width = 1024
            new_height = 256
            data = interp(np.linspace(0,num_freq_bins,new_width),
                          np.linspace(inds[0],inds[-1],new_height))
        return data

#%%
d = transform_sig(data)
plot_unlabeled_sample(d)

#%%
ss_data = []
t = 'SS'
outdir = '/Users/widemann1/Desktop/phoenix_samples/ss'
for idx, d in enumerate(zip(sig,label)):
    data,l = d
    data = np.array(data)
    l = int(l)
    print(l)
    title = '{} Sample: {:d} Count: {:d}'.format(t,idx,l)
    data = transform_sig(data)
    outfile = os.path.join(outdir,'sample_{:d}.png'.format(idx))
    plot_unlabeled_sample(data,outfile=outfile,title=title)
    ss_data.append((data,l))

with open('ss_data.pkl','wb') as fid:
    pickle.dump(ss_data,fid)
    
    
#%%
aa_data = []
with open(fn_aa,'rb') as fid:
    spectra_aa = pickle.load(fid)

sig, label, c = spectra_aa

t = 'AA'
outdir = '/Users/widemann1/Desktop/phoenix_samples/aa'
for idx, d in enumerate(zip(sig,label)):
    data,l = d
    data = np.array(data)
    l = int(l)
    print(l)
    title = '{} Sample: {:d} Count: {:d}'.format(t,idx,l)
    data = transform_sig(data)
    outfile = os.path.join(outdir,'sample_{:d}.png'.format(idx))
    plot_unlabeled_sample(data,outfile=outfile,title=title)
    aa_data.append((data,l))

with open('aa_data.pkl','wb') as fid:
    pickle.dump(aa_data,fid)
    
     
#%%
    
