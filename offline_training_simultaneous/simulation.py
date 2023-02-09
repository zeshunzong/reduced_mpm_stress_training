import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info

import torch
from typing import Optional
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os 
import math
import h5py

from util import *

'''
Simulation DataModule
'''

class SimulationDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str = "path/to/dir", batch_size: int = 32, num_workers=1, train_every_k_timestep = 1, train_every_n_points = 1):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_list = DataList(self.data_path, 1.0, train_every_k_timestep)
        assert (len(self.data_list.data_list) > 0)

        self.sim_dataset = SimulationDataset(self.data_path, self.data_list.data_list, train_every_n_points)
        self.computeStandardizeTransformation()

    def train_dataloader(self):
        return DataLoader(self.sim_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.sim_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True)
    
    def gQ_displacement(self, idx):
        data_item = self.sim_dataset[idx]
        q = data_item['q']
        return q
    
    def gQ_stress(self, idx):
        data_item = self.sim_dataset[idx]
        q = data_item['stress']
        return q

    def gX(self, idx):
        data_item = self.sim_dataset[idx]
        x = data_item['x']
        return x

    def computeMeanAndStdQ(self):

        #stress
        preprocessed_file = os.path.join(
            self.sim_dataset.data_path, 'meanandstd_stress.npy')
        if os.path.exists(preprocessed_file):
            pass
        else:
            qs = map(self.gQ_stress, range(len(self.sim_dataset)))
            qs = np.vstack(qs)
            self.mean_q_stress = np.mean(qs, axis=0)
            self.std_q_stress = np.std(qs, axis=0)
            # process stage with zero std
            for i in range(len(self.std_q_stress)):
                if self.std_q_stress[i] < 1e-12:
                    self.std_q_stress[i] = 1

        # write
            with open(preprocessed_file, 'wb') as f:
                np.save(f, self.mean_q_stress)
                np.save(f, self.std_q_stress)
        
        with open(preprocessed_file, 'rb') as f:
            self.mean_q_stress = np.load(f)
            self.std_q_stress = np.load(f)
        

        #displacement
        preprocessed_file = os.path.join(
            self.sim_dataset.data_path, 'meanandstd_q.npy')
        if os.path.exists(preprocessed_file):
            pass
        else:
            qs = map(self.gQ_displacement, range(len(self.sim_dataset)))
            qs = np.vstack(qs)
            self.mean_q_displacement = np.mean(qs, axis=0)
            self.std_q_displacement = np.std(qs, axis=0)
            # process stage with zero std
            for i in range(len(self.std_q_displacement)):
                if self.std_q_displacement[i] < 1e-12:
                    self.std_q_displacement[i] = 1

        # write
            with open(preprocessed_file, 'wb') as f:
                np.save(f, self.mean_q_displacement)
                np.save(f, self.std_q_displacement)
        
        with open(preprocessed_file, 'rb') as f:
            self.mean_q_displacement = np.load(f)
            self.std_q_displacement = np.load(f)
        

    def computeMeanAndStdX(self):
        preprocessed_file = os.path.join(
            self.sim_dataset.data_path, 'meanandstd_x.npy')
        if os.path.exists(preprocessed_file):
            pass
        else:
            xs = map(self.gX, range(len(self.sim_dataset)))
            xs = np.vstack(xs)
            self.mean_x = np.mean(xs, axis=0)
            self.std_x = np.std(xs, axis=0)

        # write
            with open(preprocessed_file, 'wb') as f:
                np.save(f, self.mean_x)
                np.save(f, self.std_x)
        
        with open(preprocessed_file, 'rb') as f:
            self.mean_x = np.load(f)
            self.std_x = np.load(f)
    
    def computeMinAndMaxX(self):
        preprocessed_file = os.path.join(
            self.sim_dataset.data_path, 'minandmax_x.npy')
        if os.path.exists(preprocessed_file):
            pass
        else:
            xs = map(self.gX, range(len(self.sim_dataset)))
            xs = np.vstack(xs)
            self.min_x = np.min(xs, axis=0)
            self.max_x = np.max(xs, axis=0)

        # write
            with open(preprocessed_file, 'wb') as f:
                np.save(f, self.min_x)
                np.save(f, self.max_x)
        
        with open(preprocessed_file, 'rb') as f:
            self.min_x = np.load(f)
            self.max_x = np.load(f)

    def computeMinAndMaxQ(self):
        # displacement
        preprocessed_file = os.path.join(
            self.sim_dataset.data_path, 'minandmax_q.npy')
        if os.path.exists(preprocessed_file):
            pass
        else:
            qs = map(self.gQ_displacement, range(len(self.sim_dataset)))
            qs = np.vstack(qs)
            self.min_q_displacement = np.min(qs, axis=0)
            self.max_q_displacement = np.max(qs, axis=0)

        # write
            with open(preprocessed_file, 'wb') as f:
                np.save(f, self.min_q_displacement)
                np.save(f, self.max_q_displacement)
        
        with open(preprocessed_file, 'rb') as f:
            self.min_q_displacement = np.load(f)
            self.max_q_displacement = np.load(f)

        # stress
        preprocessed_file = os.path.join(
            self.sim_dataset.data_path, 'minandmax_stress.npy')
        if os.path.exists(preprocessed_file):
            pass
        else:
            qs = map(self.gQ_stress, range(len(self.sim_dataset)))
            qs = np.vstack(qs)
            self.min_q_stress = np.min(qs, axis=0)
            self.max_q_stress = np.max(qs, axis=0)

        # write
            with open(preprocessed_file, 'wb') as f:
                np.save(f, self.min_q_stress)
                np.save(f, self.max_q_stress)
        
        with open(preprocessed_file, 'rb') as f:
            self.min_q_stress = np.load(f)
            self.max_q_stress = np.load(f)

    def computeStandardizeTransformation(self):
        
        self.computeMeanAndStdQ()
        self.computeMeanAndStdX()
        self.computeMinAndMaxX()
        self.computeMinAndMaxQ()
    
    def get_dataParams(self,): 

        return {'mean_q_displacement': self.mean_q_displacement, 'std_q_displacement': self.std_q_displacement, \
        'mean_q_stress': self.mean_q_stress, 'std_q_stress': self.std_q_stress, \
        'min_q_displacement': self.min_q_displacement, 'max_q_displacement': self.max_q_displacement,\
        'mean_x': self.mean_x, 'std_x': self.std_x, 'min_x': self.min_x, 'max_x': self.max_x}

    def get_dataFormat(self, ):

        # sim_dataset[i] == sim_dataset.__get_item__(i), which reads the i_th h5 file from the filelist
        example_input_array = torch.unsqueeze(self.sim_dataset[0]['encoder_input'], 0)
        [_, i_dim] = self.sim_dataset[0]['x'].shape
        [npoints, o_dim] = self.sim_dataset[0]['q'].shape
        [_, mu_dim] = self.sim_dataset[0]['mu'].shape

        data_format = {'i_dim': i_dim, 'o_dim': o_dim, 'mu_dim': mu_dim, 'npoints': npoints, 'data_path': self.data_path}

        return data_format, example_input_array


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
        stress = sim_data.stress[:]
        mu = sim_data.mu[:]
        time = sim_data.t


        if self.train_every_n_points == 1:
            pass
        else:
            num_samples = int(x.shape[0] / self.train_every_n_points)
            x = x[0:num_samples, :]
            q = q[0:num_samples, :]
            mu = mu[0:num_samples, :]


        
        x = torch.from_numpy(x).float()
        q = torch.from_numpy(q).float()
        stress = torch.from_numpy(stress).float()
        mu = torch.from_numpy(mu).float()

        encoder_input = torch.cat((q, x, mu), 1)
      
        data_item = {'filename': sim_data.filename, 'x': x,
                     'q': q, 'mu': mu, 'stress': stress,
                     'encoder_input': encoder_input, 'time': time}

        return data_item

'''
Simulation State
'''
# Assume that 'prev_label' is the encoder_prev(prev_var)
# 'q' is the current training target
class SimulationState(object):
    def __init__(self, filename, readfile=True, input_x=None, input_q=None, input_stress=None, input_t=None, input_mu = None, label=None):
        self.filename = filename
        if readfile:
            with h5py.File(self.filename, 'r') as h5_file:
                self.x = h5_file['/x'][:]
                self.x = np.array(self.x.T)
                self.q = h5_file['/q'][:]
                self.q = np.array(self.q.T)
                self.stress = h5_file['/stress'][:]
                self.stress = np.array(self.stress.T)
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
            if input_stress is None:
                print('must provide a stress if not reading from file')
                exit()
            if input_t is None:
                print('must provide a t if not reading from file')
                exit()
            if input_mu is None:
                print('must provide a mu if not reading from file')
                exit()
            self.x = input_x
            self.q = input_q
            self.stress = input_stress
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
            dset = h5_file.create_dataset("q", data=self.q.T)
            dset = h5_file.create_dataset("stress", data=self.stress.T)
            self.t = self.t.astype(np.float64)
            dset = h5_file.create_dataset("time", data=self.t)
            dset = h5_file.create_dataset("mu", data=self.mu.T)
            if self.label is not None:
                label = self.label.reshape(-1, 1)
                label = label.astype(np.float64)
                dset = h5_file.create_dataset("label", data=label)

'''
Obj Loader
'''

# from: https://inareous.github.io/posts/opening-obj-using-py
# also checkout: https://pypi.org/project/PyWavefront/

class ObjLoader(object):
    def __init__(self, fileName=None):
        self.vertices = []
        self.faces = []
        ##
        if fileName:
            try:
                f = open(fileName)
                for line in f:
                    if line[:2] == "v ":
                        index1 = line.find(" ") + 1
                        index2 = line.find(" ", index1 + 1)
                        index3 = line.find(" ", index2 + 1)

                        vertex = (float(line[index1:index2]), float(
                            line[index2:index3]), float(line[index3:-1]))
                        self.vertices.append(vertex)

                    elif line[0] == "f":
                        string = line.replace("//", "/")
                        ##
                        i = string.find(" ") + 1
                        face = []
                        for item in range(string.count(" ")):
                            if string.find(" ", i) == -1:
                                face.append(string[i:-1])
                                break
                            face.append(string[i:string.find(" ", i)])
                            i = string.find(" ", i) + 1
                        ##
                        self.faces.append(tuple(face))

                f.close()
            except IOError:
                print(".obj file not found.")

    def export(self, filename):
        f = open(filename, "w")
        f.write("g ")
        f.write("\n")
        for vertex in self.vertices:
            line = "v " + " " + \
                str(vertex[0]) + " " + \
                str(vertex[1]) + " " + str(vertex[2])
            f.write(line)
            f.write("\n")
        f.write("g ")
        f.write("\n")
        for face in self.faces:
            line = "f " + " " + \
                str(face[0]) + " " + \
                str(face[1]) + " " + str(face[2])
            f.write(line)
            f.write("\n")
        f.close()
