
import os, sys
from time import time, sleep
import pickle
import numpy as np
from nb_generate_wave_eqn_data import *
import pickle
from time import time


outdir = 'samples'
if not os.path.exists(outdir):
    os.makedirs(outdir)

num_sources_list = [130] #np.concatenate([[1],np.arange(10,370,10)])

c = 343 # m/s
f0 = 1002 # Hz
d_Hz = .1 # Hz
fs = np.round(8.1*f0)
dt = 1/fs # we have to sample at twice f0 
nt = 30000 #10*60*fs # 10 minutes of data
dx = .2
num_keep = 8192

for num_sources in num_sources_list:
    t0 = time()
    u, prev_u, source_locs, f = generate_input(num_sources=num_sources,nt=nt,f0=f0,d_Hz=d_Hz)
    s = wave_steady_state(u,prev_u,nt,dt,dx,c=c,num_keep=num_keep)
    '''
    s_flat = []
    for i in range(100):
        for j in range(100):
            s_flat.append((np.squeeze(s[:,i,j]),num_sources))
    '''
    fn = os.path.join(outdir,'sample_{:d}_sources_fs_{:d}.pkl'.format(num_sources,int(fs)))
    with open(fn,'wb') as fid:
        pickle.dump(s,fid)
    print('run-time {:.2f}'.format(time() - t0))
    
