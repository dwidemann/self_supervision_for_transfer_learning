import os, sys
import pandas as pd
import numpy as np
import pickle

direc_name = '/gsceph/adapd/acoustic/acoustic_ffts_10min'
fn = '/data/acoustic_tmp/unlabeled_acoustic_10min/labels/unique_mapped_acoustic_labels.csv'

data = pd.read_csv(fn)
AA = []
SS = []

def int2class(i):
    c = np.nan
    if i == 0:
        c = 0
    elif (i >= 1 and i <= 10):
        c = 1
    elif (i >= 60 and i<= 70):
        c = 2
    elif (i > 100):
        c = 3
    return c

for idx in range(data.shape[0]):
    r = data.iloc[idx]
    if 'A/A' in r[1]:
        s = int(r[1].replace('A/A',''))
        label = int2class(s)
        f = os.path.join(direc_name, r[0])
        AA.append((f,label))
    elif 'S/S' in r[1]:
        s = int(r[1].replace('S/S',''))
        label = int2class(s)
        f = os.path.join(direc_name, r[0])
        SS.append((f,label))

outdir = '/gsceph/adapd/acoustic/AA_10'
if not os.path.exists(outdir):
    os.makedirs(outdir)
with open(os.path.join(outdir,'train.pkl'), 'wb') as fid:
    pickle.dump(AA,fid)

outdir = '/gsceph/adapd/acoustic/SS_10'
if not os.path.exists(outdir):
    os.makedirs(outdir)
with open(os.path.join(outdir,'train.pkl'), 'wb') as fid:
    pickle.dump(AA,fid)

