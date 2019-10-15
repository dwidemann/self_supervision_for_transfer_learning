#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:55:04 2019

@author: widemann1
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pickle
from scipy.interpolate import interp2d
import multiprocessing as mp
from time import time, sleep

#%%
def load_pkl_file(fn,interp=True, normalize=True):
    with open(fn,'rb') as fid:
        data = pickle.load(fid)
        freqs = data['freqs']
        R = data['sig']

        sig = np.abs(R)
        A = np.abs(sig)
        A = np.log10(A + 1)
        
        if (normalize == True):
            print('normalizing the acoustic data.')
            sys.exit(1)
            
            mu = A.mean(axis=1).reshape(A.shape[0],1)
            std = A.std(axis=1).reshape(A.shape[0],1)

            data = (A-mu)/std
        else:
            data = A

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
    return data

def mp_worker(d):
    fn, rescale, outdir, normalize = d
    t0 = time()
    try: 
        plot_unlabeled_sample(fn,rescale,outdir, normalize)
        print("sample {} runtime: {:0.2f}".format(fn,time()-t0))
    except Exception as e:
        print('error {}: reason: {}'.format(fn, e))


def plot_unlabeled_sample(fn,rescale=True,outfile=None, normalize=True):
    A_normalized = load_pkl_file(fn,rescale, normalize)

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
    
    plt.close(fig)
    #%%
    # fig, axs = plt.subplots(0,0)
    # plt.imshow(A_normalized)
    # #axs.set_aspect('equal', 'box')
    # fig.tight_layout()
    # # plt.gca().set_aspect('scaled', adjustable='box')
    # # plt.gca().set_limits('tight', adjustable='box')
    # plt.show()

if __name__ == '__main__':
    def mp_handler(num_cpus=30):
        p = mp.Pool(num_cpus)
        p.map(mp_worker, queue)

    fn = sys.argv[1]
    rescale = 1
    if len(sys.argv) > 2:
        rescale = int(sys.argv[2])
    
    normalize = True
    if len(sys.argv) > 3:
        normalize = int(sys.argv[3])
        
    if os.path.isdir(fn):
        fns = glob(os.path.join(fn, '*.pkl') )
    else:
        fns = [fn]
    outdir='../images/unlabeled_acoustic/'
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)
   
    idx = fn.rfind('/') + 1 #get index of where the file name starts
    queue = []
    for f in fns:
        queue.append((f, rescale, outdir + str(f[idx:].replace('pkl', 'png')), normalize) )
    
    mp_handler()
    
