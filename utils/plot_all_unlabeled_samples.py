#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 09:56:50 2019

@author: widemann1
"""

import os, sys
from plot_unlabeled_sample import plot_unlabeled_sample
import pickle

#%%
dataFile = '/data/ADAPD/whole_apron_fft/apron_labels/all_1min_samples.pkl'
rescale = True
image_direc = '~/adapd/images/'

#%%
with open(dataFile,'rb') as fid:
    data = pickle.load(fid)

#%%

if not os.path.exists(image_direc):
    os.makedirs(image_direc)
    
for d in data:
    print(d)
    fn = d[0]
    sp = fn.split('/')
    outdir = str(int(d[4])) 
    
    dout = os.path.join(image_direc,outdir)
    if not os.path.exist(dout):
        os.makedirs(dout)
    outname = os.path.join(sp[-2],sp[-1].replace('.pkl','.png'))
    outfile = os.path.join(dout,outname)
    plot_unlabeled_sample(fn,rescale,outfile)

