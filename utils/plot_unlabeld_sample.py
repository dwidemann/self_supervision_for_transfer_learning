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
#%%
def load_pkl_file(fn):
    with open(fn,'rb') as fid:
        data = pickle.load(fid)
        freqs, R = zip(*data)
        L = []
        for i in R:
            L.append(len(i))
        m = np.min(L)
        data = []
        for r in R:
            data.append(r[:m]) # make sure each row has the same number of entries

        data = np.array(data)
        inds = np.arange(2,101,6)
        sig = data[inds,:]
        A = np.abs(sig)
        A = np.log10(A)
        
        mu = A.mean(axis=1).reshape(len(inds),1)
        std = A.std(axis=1).reshape(len(inds),1)
        
        data = (A-mu)/std
    return data

def plot_unlabeled_sample(fn):
    A_normalized = load_pkl_file(fn)

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
    plot_unlabeled_sample(fn)
    