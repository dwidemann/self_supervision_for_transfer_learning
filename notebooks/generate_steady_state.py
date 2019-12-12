
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

c = 1 # m/s
f0 = 1002 # Hz
d_Hz = .1 #400 # Hz
dt = .0001
fs = 1/dt
nt = 2**14 + 1 #10*60*fs # 10 minutes of data
dx = .0002
num_keep = 2**14
insync = True
leftOfF0 = False #True

for num_sources in num_sources_list:
    t0 = time()
    u, prev_u, source_locs, f = generate_input(num_sources=num_sources,
                                               max_u0=1.,
                                               f0=f0,
                                               d_Hz=d_Hz,nt=nt,dt=dt,
                                               insync=insync,
                                               leftOfF0=leftOfF0)


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
    
