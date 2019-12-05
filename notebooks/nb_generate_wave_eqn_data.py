
        #################################################
        ### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
        #################################################
        # file to edit: dev_nb/generate_wave_eqn_data.ipynb

import os, sys
from time import time, sleep
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as manimati
from matplotlib import animation, rc
from numpy.fft import rfft
from IPython.display import HTML

def generate_input(num_sources=128,max_u0=1.,grid_size=100,f0=2,d_Hz=.1,nt=200,dt=.1,insync=False):
    # len(v)*len(w) = 360
    v = np.arange(5,grid_size-5,2)
    w = np.arange(5,20,2)
    x,y = np.meshgrid(v,w)
    t = dt*np.arange(nt)
    potential_sources = np.array(list(zip(x.flatten(),y.flatten())))
    inds = np.random.choice(len(potential_sources),size=num_sources,replace=False)
    f = torch.zeros((nt,grid_size,grid_size),requires_grad=True)
    source_locs = []
    freqs = []
    for idx in inds:
        i,j = potential_sources[idx]
        source_locs.append(potential_sources[i])
        f0_samp = d_Hz*np.random.randn() + f0
        freqs.append(f0_samp)
        phase = 0
        if not insync:
            phase = 2*np.pi*np.random.rand()
        f[:,j,i] = torch.tensor(max_u0*np.sin(2*np.pi*(f0_samp*t)+phase))
        #plt.plot(t,f[:,j,i].detach().numpy())
    prev_u = torch.zeros((grid_size,grid_size),requires_grad=True)
    u, prev_u = Variable(f, requires_grad=True), Variable(prev_u, requires_grad=True)
    return u, prev_u, source_locs, freqs

def plot_image(f, title='wave amplitude'):
    # plt.ion()
    u_mx = f.max()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(title)
    cmap = plt.cm.ocean
    img = ax.imshow(f.data, cmap=cmap, vmin=-u_mx, vmax=u_mx)

    fig.colorbar(img, orientation='vertical')
    plt.show()
    return img, fig

def Laplacian(u):
    u = u.unsqueeze(0).unsqueeze(0)
    L = np.array([[.5, 1, .5], [1, -6., 1], [.5, 1, .5]], 'float32')
    pad = (L.shape[0]-1)//2
    conv = nn.Conv2d(1, 1, L.shape[0], 1, bias=False, padding=(pad, pad))
    conv.weight.data = torch.tensor(L).unsqueeze(0).unsqueeze(0)
    return conv(u).squeeze(0).squeeze(0)

def run_wave_forward(f,prev_u,nt,dt,dx,c):
    u = f[0,:,:]
    soln = [u]
    DT_DX_SQ = c*(dt/dx)**2
    for i in range(nt-1):
        next_u = DT_DX_SQ*Laplacian(u) + 2*u - prev_u + dt*f[i,:,:]
        prev_u = u
        u = next_u
        soln.append(u)
    soln = np.array([i.detach().numpy() for i in soln])
    return soln

def wave_steady_state(f,prev_u,nt,dt,dx,c,num_keep):
    u = f[0,:,:]
    soln = []
    DT_DX_SQ = c*(dt/dx)**2
    for i in range(nt-1):
        next_u = DT_DX_SQ*Laplacian(u) + 2*u - prev_u + dt*f[i,:,:]
        prev_u = u
        u = next_u
        if i >= nt - num_keep -1 :
            soln.append(u)
    soln = np.array([i.detach().numpy() for i in soln])
    return soln


def create_movie(s,outfile='wave_equation.mp4'):
    ti = 0
    title = 'wave amplitude'
    u_mx = np.max(np.abs(s))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(title)
    cmap = plt.cm.ocean
    img = ax.imshow(s[0], cmap=cmap, vmin=-u_mx, vmax=u_mx)
    fig.colorbar(img, orientation='vertical')
    #plt.show()

    # initialization function: plot the background of each frame
    def init():
        img = ax.imshow(s[0], cmap=cmap, vmin=-u_mx, vmax=u_mx)
        return (fig,)

    # animation function. This is called sequentially
    def animate(i):
        img = ax.imshow(s[i], cmap=cmap, vmin=-u_mx, vmax=u_mx)
        return (fig,)

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(s), interval=20, blit=True)
    anim.save(outfile, fps=30, extra_args=['-vcodec', 'libx264'])

def generate_wave_data(num_sources=10,f0=1.,d_Hz=.1,c=1,nt=200,dt=.1,dx=.2,insync=False):
    u, prev_u, source_locs, f = generate_input(num_sources=num_sources,
                                               max_u0=1.,
                                               f0=f0,
                                               d_Hz=d_Hz,nt=nt,dt=dt,insync=insync)
    soln = run_wave_forward(u,prev_u,nt,dt,dx,c=c)
    return soln

def plot_fft(num_sources=130,f0=1,dt=.1,nt=200,c=1,dx=.2,insync=True,i=50,j=50):
    s = generate_wave_data(num_sources=num_sources,f0=f0,dt=dt,nt=nt,c=c,dx=dx,insync=True)
    s0 = s[:,i,j].squeeze()
    t = dt*np.arange(len(s0))
    plt.plot(t,s0)
    nfft = len(s0)
    plt.show()
    S = rfft(s0, nfft)
    fs = 1/dt
    freqs = np.linspace(0,fs/2,len(S))
    plt.plot(freqs,np.abs(S))
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.show()

from torch.utils.data import Dataset, DataLoader
class WaveDataset(Dataset):
    def __init__(self, n_sims=100,transform=None):

        self.sim_params = []
        data = []
        for k in range(n_sims):
            d = dict()
            d['num_sources'] = 1 #np.random.randint(5,16)
            #self.sim_params.append(d)
            X = generate_wave_data(**d)
            X = [xi.detach().numpy() for xi in X]
            data.append(X)
        data = [item for sublist in data for item in sublist]
        # collects two frames as input and predicts the third
        # frame as output. This is an inefficient way to
        # store data.
        # TODO: Store data smarter
        stacked_frames = []
        for idx,d in enumerate(data[:-3]):
            X = np.stack((d,data[idx+1]))
            y = data[idx+2]
            stacked_frames.append((X,y))

        self.data = stacked_frames
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X,y = self.data[idx]
        if self.transform:
            X = self.transform(X)
        return X,y
