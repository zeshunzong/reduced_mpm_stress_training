import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
import numpy as np
import os 
import math
import h5py
from ObjLoader import *
from util import *

'''
Simulation Dataset
'''

class SimulationDataset(Dataset):
    def __init__(self, data_path, data_list, train_every_n_points=1):
        self.data_list = data_list
        self.data_path = data_path
        self.train_every_n_points = train_every_n_points

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        filename = self.data_list[idx]
        sim_data = SimulationState(filename)
        x = sim_data.x[:]
        q = sim_data.q[:]
        mu = sim_data.mu[:]
        prev_label = sim_data.prev_label[:]
        time = sim_data.t


        if self.train_every_n_points == 1:
            pass
        else:
            num_samples = int(x.shape[0] / self.train_every_n_points)
            x = x[0:num_samples, :]
            q = q[0:num_samples, :]
            mu = mu[0:num_samples, :]
            prev_label = prev_label[0:num_samples, :]


        
        x = torch.from_numpy(x).float()
        q = torch.from_numpy(q).float()
        mu = torch.from_numpy(mu).float()
        prev_label = torch.from_numpy(prev_label).float()


        encoder_input = torch.cat((prev_label, x, mu), 1)
      
        data_item = {'filename': sim_data.filename, 'x': x,
                     'q': q, 'prev_label': prev_label, 'mu': mu,
                     'encoder_input': encoder_input, 'time': time}

        return data_item

'''
Simulation State
'''
# Assume that 'prev_label' is the encoder_prev(prev_var)
# 'q' is the current training target
class SimulationState(object):
    def __init__(self, filename, readfile=True, input_x=None, input_q=None, input_t=None, input_mu = None, label=None):
        self.filename = filename
        if readfile:
            with h5py.File(self.filename, 'r') as h5_file:
                self.x = h5_file['/x'][:]
                self.x = np.array(self.x.T)
                self.q = h5_file['/stress'][:]
                self.q = np.array(self.q.T)
                self.prev_label = h5_file['/prev_label'][:]
                self.prev_label = np.array(self.prev_label.T)
                self.mu = h5_file['/mu'][:]
                self.mu = np.array(self.mu.T)
                self.t = h5_file['/time'][0][0]
              
        else:
            if input_x is None:
                print('must provide a x if not reading from file')
                exit()
            if input_q is None:
                print('must provide a q if not reading from file')
                exit()
            if input_t is None:
                print('must provide a t if not reading from file')
                exit()
            if input_mu is None:
                print('must provide a mu if not reading from file')
                exit()
            self.x = input_x
            self.q = input_q
            self.t = input_t
            self.label = label
            self.mu = input_mu
    
    def write_to_file(self, filename=None):
        if filename:
            self.filename = filename
        print('writng sim state: ', self.filename)
        dirname = os.path.dirname(self.filename)
        os.umask(0)
        os.makedirs(dirname, 0o777, exist_ok=True)
        with h5py.File(self.filename, 'w') as h5_file:
            dset = h5_file.create_dataset("x", data=self.x.T)
            dset = h5_file.create_dataset("stress", data=self.q.T)
            self.t = self.t.astype(np.float64)
            dset = h5_file.create_dataset("time", data=self.t)
            dset = h5_file.create_dataset("mu", data=self.mu.T)
            if self.label is not None:
                label = self.label.reshape(-1, 1)
                label = label.astype(np.float64)
                dset = h5_file.create_dataset("label", data=label)
        



