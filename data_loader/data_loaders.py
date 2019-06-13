from torchvision import datasets, transforms
from base import BaseDataLoader
import os, sys
from glob import glob
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class ApronDataLoader(BaseDataLoader):
    '''
    DataLoader for the unlabeled Apron data. 
    '''
    # TODO: Modify code so that it works for the few labeled Apron examples too.  
    def __init__(self, data_dir='/data/ADAPD/whole_apron_fft/apron_1m_unlabeled', batch_size=16, shuffle=True, 
                validation_split=0.0, num_workers=2, training=True,unlabeled=True):
        trsfm = transforms.Compose([
            transforms.ToTensor()
        ])
        self.data_dir = data_dir
        d = os.path.join(data_dir,'*','*.pkl')
        pkl_files = glob(d,recursive=True)
        def _load_pkl_file(fn):
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
                data = np.array(data).astype('complex64')
                # normalization will happen before the first layer with a batch_norm
                #normed_data = data/np.linalg.norm(data)
                #out = np.array([np.real(normed_data), np.imag(normed_data)])
                out = np.array([np.real(data), np.imag(data)])
                if unlabeled: # this is a hack. Not sure the correct way to do this. 
                    out = (out,np.nan)
                return out

        self.data = list(map(lambda x: _load_pkl_file(x),pkl_files[:32]))
        #self.dataset = DataLoader(data,batch_size,shuffle,num_workers)
        super(ApronDataLoader, self).__init__(self.data, batch_size, shuffle, validation_split, num_workers)


class ApronDataset(Dataset):
    def __init__(self, pkl_file, transform=None):

        with open(pkl_file,'rb') as fid:
            self.data = pickle.load(fid)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def load_pkl_file(self,fn):
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
            data = np.array(data).astype('complex64')
            out = np.array([np.real(data), np.imag(data)])
            return out

    def __getitem__(self, idx):
        fn,y = self.data[idx]
        X = self.load_pkl_file(fn)
        if self.transform:
            X = self.transform(X)
        return X,y



class ApronDataLoaderGenerator(BaseDataLoader):
    '''
    DataLoader for the unlabeled Apron data. 
    '''
    # TODO: Modify code so that it works for the few labeled Apron examples too.  
    def __init__(self, data_dir='/data/ADAPD/whole_apron_fft/train.pkl', batch_size=16, shuffle=True, 
                validation_split=0.0, num_workers=2, training=True,unlabeled=True):
        trsfm = transforms.Compose([
            transforms.ToTensor()
        ])
        self.data_dir = data_dir
        self.data = ApronDataset(data_dir)
        super(ApronDataLoaderGenerator, self).__init__(self.data, batch_size, shuffle, validation_split, num_workers)


