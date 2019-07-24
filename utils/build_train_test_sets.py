#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 12:01:37 2019

@author: widemann1
"""
import os, sys
import pickle
import numpy as np
#%%
def mapSampLabel(samp):
    x,y = samp
    y = int(y)
    if y == 0:
        label = 0
    elif y == 1:
        label = 1
    elif y == 64:
        label = 2
    elif y == 128:
        label = 3
    #label = np.array(label).astype('int32')
    samp = (x,label)
    return samp

def build_train_test(test_idx,data_file,outdir):
    with open(data_file,'rb') as fid:
        data = pickle.load(fid)
        test = []
        train = []
        for idx, samp in enumerate(data):
            samp = mapSampLabel(samp)
            if idx == test_idx:
                test.append(samp)
            else:
                train.append(samp)
    
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(os.path.join(outdir,'train.pkl'),'wb') as fid:
        pickle.dump(train,fid)

    with open(os.path.join(outdir,'test.pkl'),'wb') as fid:
        pickle.dump(test,fid)

if __name__ == '__main__':
    test_idx = int(sys.argv[1])
    data_file = sys.argv[2]
    outdir = sys.argv[3]
    build_train_test(test_idx,data_file,outdir)