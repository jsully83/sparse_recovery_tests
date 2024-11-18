from collections.abc import Generator
from pathlib import Path

import numpy as np
import torch
import yaml

class Dyan(Generator):
    def __init__(self, config_filename, dictionary_filename=None) -> None:
        
        # Check if a GPU is available and set the device accordingly
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data_path = Path.cwd().joinpath('data')
        self.config_filename = Path(config_filename)
        
        self.configuration()
        
        if dictionary_filename is None:
            if hasattr(self, 'dictionary_filename'):
                self.dictionary_path = self.data_path.joinpath(self.dictionary_filename)
            else:
                raise ValueError("Dictionary filename must be provided either during instantiation or in the configuration file.")
        else:
            self.dictionary_path = self.data_path.joinpath(dictionary_filename)
    
        self.load_raw_dict()
        self.generate_dictionary()
    
    def configuration(self):
        if self.config_filename.suffix != '.yaml':
            self.config_filename = self.config_filename.with_suffix('.yaml')

        config_files = list(Path.cwd().rglob(self.config_filename.name))
        if len(config_files) > 1:
            raise FileExistsError(f"Multiple configuration files found: {config_files}")
        elif len(config_files) == 1:
            self.config_filename = config_files[0]
        else:
            raise FileNotFoundError(f'{self.config_filename} not found.')

        with open(self.config_filename, 'r') as f:
            self.config = yaml.safe_load(f)

        required_keys = ['T', 'Np', 'n_poles', 'unity_poles', 'normalized_output']
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Required key '{key}' not found in configuration file.")

        for key, value in self.config.items():
            setattr(self, key, value)

    
    def load_raw_dict(self):
        suf = self.dictionary_path.suffix
        if suf == '.pth':
            self.raw_data = torch.load(self.dictionary_path.as_posix(), map_location=self.device)['state_dict']
            self.r = self.raw_data['sparseCoding.rr']
            self.theta = self.raw_data['sparseCoding.theta']
        elif suf == '.npy':
            self.raw_data = np.load(self.dictionary_path.as_posix())
            self.r = self.raw_data[0,:]
            self.theta = self.raw_data[1,:]
        else:
            raise ValueError
        
    def generate_dictionary(self):
        self.dictionary = np.zeros([self.T,self.Np])
        
        for i in range(0,self.T):
            r_pow = np.power(self.r,i)
            self.dictionary[i,:] = np.concatenate([[1], r_pow * np.cos(i*self.theta),
                                     r_pow * np.sin(i*self.theta)])
        
        self.normalize_matrix(self.dictionary)
            
            
    def normalize_matrix(self, arr):
        column_norm = np.linalg.norm(arr, ord=2.0, axis=0, keepdims=True)
        self.dictionary_norm = arr / column_norm
                
    def create_random_coeff(self):
        self.C = np.zeros(self.Np)
        if self.unity_poles:
            c = np.ones(self.n_poles)
        else:
            c = np.random.uniform(-1,1, self.n_poles)
            
        atoms = 0
        while np.unique(atoms).shape[0] < self.n_poles: # make sure we get enough unique poles.
            atoms = np.random.randint(0, self.Np, self.n_poles)

        for i in range(self.n_poles): # distribute 
            self.C[atoms[i]] = c[i]
            
    
    def __iter__(self):
        return self

    def __next__(self):
        self.create_random_coeff()
        y = np.matmul(self.dictionary, self.C)
        
        if self.normalized_output:
            y = y / np.linalg.norm(y, ord=2)
            
        return y, self.C

    def send(self, value):
        self.C = np.asarray(value)
        return self.__next__()

    def throw(self, typ=None, val=None, tb=None):
        raise StopIteration

