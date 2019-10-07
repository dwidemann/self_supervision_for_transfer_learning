#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:55:04 2019

@author: widemann1
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp2d

#%%
def load_pkl_file(fn,interp=True):
    with open(fn,'rb') as fid:
        data = pickle.load(fid)
        freqs = data['freqs']
        R = data['sig']

        sig = np.abs(R)
        A = np.abs(sig)
        A = np.log10(A)

        mu = A.mean(axis=1).reshape(A.shape[0],1)
        std = A.std(axis=1).reshape(A.shape[0],1)

        data = (A-mu)/std

        if interp:
            inds = range(A.shape[0])
            num_freq_bins = data.shape[1]
            interp = interp2d(range(num_freq_bins),
                              range(A.shape[0]),
                              data,kind='linear')

            new_width = 1024
            new_height = 256
            data = interp(np.linspace(0,num_freq_bins,new_width),
                          np.linspace(inds[0],inds[-1],new_height))

        data = data.astype('float32')
        print(data.shape)
    return data

def plot_unlabeled_sample(fn,rescale=True,outfile=None):
    A_normalized = load_pkl_file(fn,rescale)

    #plt.style.use('ggplot')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(A_normalized)
    ratio = 1.0
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    # the abs method is used to make sure that all numbers are positive
    # because x and y axis of an axes maybe inversed.
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
    fig.colorbar(im)
    # or we can utilise the get_data_ratio method which is more concise
    # ax.set_aspect(1.0/ax.get_data_ratio()*ratio)
    if outfile:
        plt.savefig(fname=outfile,dpi=300,format='png')
    else:
        plt.show()
    #%%
    # fig, axs = plt.subplots(0,0)
    # plt.imshow(A_normalized)
    # #axs.set_aspect('equal', 'box')
    # fig.tight_layout()
    # # plt.gca().set_aspect('scaled', adjustable='box')
    # # plt.gca().set_limits('tight', adjustable='box')
    # plt.show()

if __name__ == '__main__':
    fn = sys.argv[1]
    rescale = 1
    if len(sys.argv) > 2:
        rescale = int(sys.argv[2])
    plot_unlabeled_sample(fn,rescale)
    
    
    
    
    
