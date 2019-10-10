from torchvision import datasets, transforms
from base import BaseDataLoader
import os, sys
from glob import glob
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import make_classification
from scipy.interpolate import interp2d


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


class AcousticDataset(Dataset):
    def __init__(self, pkl_file, transform=None):
        with open(pkl_file,'rb') as fid:
            self.data = pickle.load(fid)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __len__(self):
        return len(self.data)

    def load_pkl_file(self,fn,interp=True):
        with open(fn,'rb') as fid:
            data = pickle.load(fid)
            freqs = data['freqs']
            R = data['sig']

            A = np.abs(R)
            A = np.log10(A + 1)
            
            mu = A.mean(axis=1).reshape(A.shape[0],1)
            std = A.std(axis=1).reshape(A.shape[0],1)
            eps = 1e-6
            data = (A-mu)/(std + eps)
            
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
            if len(data.shape) == 2:
                data = np.expand_dims(data,axis=0)
            print(data.shape)
        return data

    def __getitem__(self, idx):
        fn  = self.data[idx]
        # get the labels later. 
        #fn,y = self.data[idx]
        X = self.load_pkl_file(fn,interp=True)
        if self.transform:
            X = self.transform(X)
        return X

class AcousticDataLoader(BaseDataLoader):
    '''
    DataLoader for the unlabeled Apron data. 
    '''
    # TODO: Modify code so that it works for the few labeled Apron examples too.  
    def __init__(self, data_dir='/gsceph/adapd/acoustic/train_test_split', batch_size=2, shuffle=True, 
                validation_split=0.0, num_workers=2, training=True,unlabeled=True):
        trsfm = transforms.Compose([
            transforms.ToTensor()
        ])
        self.data_dir = data_dir
        self.training = training
        if self.training:
            pkl_file = os.path.join(data_dir,'train.pkl')
        else:
            pkl_file = os.path.join(data_dir,'test.pkl')
        self.pkl_file = pkl_file
        self.data = AcousticDataset(pkl_file)
        super(AcousticDataLoader, self).__init__(self.data, batch_size, shuffle, validation_split, num_workers)




class ApronDataset(Dataset):
    def __init__(self, pkl_file, transform=None):

        with open(pkl_file,'rb') as fid:
            self.data = pickle.load(fid)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def load_pkl_file(self,fn,interp=True):
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
            
            if interp:
                num_freq_bins = 1342
                interp = interp2d(range(num_freq_bins),
                                  inds,
                                  data,kind='linear')
                new_width = 1024
                new_height = 256
                data = interp(np.linspace(0,num_freq_bins,new_width),
                              np.linspace(inds[0],inds[-1],new_height))
            data = data.astype('float32')
            if len(data.shape) == 2:
                data = np.expand_dims(data,axis=0)
        return data
    '''
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

            inds = np.arange(2,101,6)
            sig = data[:,inds,:]
            A = np.sqrt(sig[0]**2 + sig[1]**2)
            A = np.log10(A)
            
            mu = A.mean(axis=1).reshape(len(inds),1)
            std = A.std(axis=1).reshape(len(inds),1)
            
            data = (A-mu)/std
            data = np.array(data).astype('complex64')
            out = np.array([np.real(data), np.imag(data)])
            return out
    '''

    def __getitem__(self, idx):
        fn,y = self.data[idx]
        X = self.load_pkl_file(fn,interp=True)
        if self.transform:
            X = self.transform(X)
        return X,y



class ApronDataLoaderGenerator(BaseDataLoader):
    '''
    DataLoader for the unlabeled Apron data. 
    '''
    # TODO: Modify code so that it works for the few labeled Apron examples too.  
    def __init__(self, data_dir='/data/ADAPD/whole_apron_fft', batch_size=16, shuffle=True, 
                validation_split=0.0, num_workers=2, training=True,unlabeled=True):
        trsfm = transforms.Compose([
            transforms.ToTensor()
        ])
        self.data_dir = data_dir
        self.training = training
        if self.training:
            pkl_file = os.path.join(data_dir,'train.pkl')
        else:
            pkl_file = os.path.join(data_dir,'test.pkl')
        self.pkl_file = pkl_file
        self.data = ApronDataset(pkl_file)
        super(ApronDataLoaderGenerator, self).__init__(self.data, batch_size, shuffle, validation_split, num_workers)


class SyntheticApronDataset(Dataset):
    def __init__(self, n_samples=10000,n_classes=4):
        self.n_classes = n_classes
        self.n_samples = n_samples
        X, y = make_classification(n_features=2*100*1342, n_redundant=0, 
                           n_informative=1000,
                           random_state=1, n_clusters_per_class=1,
                           n_samples=n_samples,
                           n_classes=n_classes)

        X = list(map(lambda x: x.reshape(2,100,1342).astype('float32'),X))

        self.data = list(zip(X,y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X,y = self.data[idx]
        return X,y
    
    
    
class SyntheticApron(BaseDataLoader):
    '''
    DataLoader for the synthetic Apron data. 
    '''
    # TODO: Modify code so that it works for the few labeled Apron examples too.  
    def __init__(self, n_samples=10000,n_classes=4, batch_size=16, shuffle=True, 
                validation_split=0.1, num_workers=2, training=True):

        self.n_samples = n_samples
        self.n_classes = n_classes
        self.data = SyntheticApronDataset(n_samples,n_classes)
        super(SyntheticApron, self).__init__(self.data, batch_size, 
             shuffle, validation_split, num_workers)
