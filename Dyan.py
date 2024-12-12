
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange
import yaml

from utils import get_Drr_Dtheta

class BaseConfig(nn.Module):
    def __init__(self, config_filename=None):
        super().__init__()
        self.required_keys = ['dictionary_filename', 'normalize_dict',
                              'batch_size', 'T', 'Np', 'joints', 'eps',
                              'fista_iter', 'fista_lambd', 'fista_reweighted', 'fista_tol']
        
        self.dictionary_filename = None
        self.normalize_dict = None
        
        self.batch_size = None
        self.T = None
        self.Np = None
        self.joints = None
        self.eps = 1e-6
        self.tol = 1e-5
        self.fista_iter = None
        self.fista_lambd = None
        self.fista_reweighted = None
        self.fista_tol = None
        
        # self.zero = nn.Buffer(torch.Tensor([self.eps]))
        # self.one = nn.Buffer(torch.Tensor([1.0 + self.eps]))
        
        if config_filename is not None:
            self.config_filename = Path(config_filename)
        else:
            raise ValueError("A configuration filename must be provided.")
        
        # load configuration
        self.load_configuration()

    def load_configuration(self, ):
        if self.config_filename.suffix != '.yaml':
            self.config_filename = self.config_filename.with_suffix('.yaml')
        
        config_path = Path.cwd().joinpath('config', self.config_filename.name)
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                    
            for key in self.required_keys:
                if key not in self.config:
                    print(f"Required key '{key}' not found in configuration file. Using default value of {getattr(self, key)}")
            
            for key, value in self.config.items():
                setattr(self, key, value)
            
            # print(f"Loaded configuration from {config_path}")

class FISTA():
    def __init__(self, device, lambd, reweighted, n_iter, tol, eps, normalize_dict):
        # super(FISTA, self).__init__(config_filename)
        self.device = device
        self.lambda_init = lambd
        self.reweighted = reweighted
        self.n_iter = n_iter
        self.tol = tol
        self.eps = eps
        self.normalize_dict = normalize_dict
    
        
    def solve_fista(self, Y, D):
        
        self.check_set_dims(Y, D)
        
        self.C_curr = torch.zeros([self.batch_size, self.Np, self.joints]).to(self.device)
        self.C_prev = torch.zeros([self.batch_size, self.Np, self.joints]).to(self.device)
        # self.t_prev = 1.0

        # precompute some mats
        Dt = D.transpose(0,1)
        self.DtD = torch.matmul(Dt, D).to(self.device) # [Np,Np]
        self.DtY = torch.matmul(Dt, Y).to(self.device) # [Batch, Np, joints]
        # self.L = 1.0 / torch.linalg.matrix_norm(self.DtD, ord=2)
        self.L = 1.0 / torch.norm(self.DtD)
        self.L = self.L.to(self.device)
        
        self.lambda_ = self.lambda_init
        
        # with torch.no_grad():
        if self.reweighted:
            for i in range(2):
                C = self.converge()
                self.weight_lambda()
        else:
            C = self.converge()
    
        return C
    
    def converge(self):
        t_prev = 1.0
        for i in range(self.n_iter):
            # print(f'i: {i} self.DtD {self.DtD.shape} self.C_next {self.C_next.device} self.C_curr {self.C_curr.device} self.C_prev {self.C_prev.shape} L {self.L.device} DtY {self.DtY.shape}')
            # calculate the gradient and soft thresholding step
            gradient = torch.matmul(self.DtD, self.C_prev) - self.DtY # [Np, Np] * [B, Np, joints] = [B, Np, joints] - [B, Np, joints]
            descent = self.C_prev - self.L*gradient # [B, Np, joints] - L*[B, Np, joints]
            C_next = self.soft_thresholding(descent, self.L*self.lambda_)
            # print(f'gradient {gradient.device} descent {descent.device}')
            # exit if it's close
            if torch.linalg.norm(C_next - self.C_curr) < self.tol:
                break

            # find the updated momentum constant
            t_next = (1 + (1 + 4 * t_prev**2)**0.5) / 2
            t_const = (t_prev - 1) / t_next 
            t_prev = t_next
            # self.t_prev.copy_(t_next)
            
            # momentum step
            self.C_prev = torch.clone(C_next + t_const * (C_next - self.C_curr))

            self.C_curr = torch.clone(C_next)
            # C_next.detach()

        return self.C_curr
    
    def weight_lambda(self):
        self.weights = 1 / (torch.abs(self.C_curr) + self.eps)
        self.weights = self.weights / torch.linalg.vector_norm(self.weights, dim=1, keepdims=True)
        self.lambda_ = self.lambda_ * self.weights * self.Np
                    
    def soft_thresholding(self, arr, tau):
        a = torch.sign(arr) * torch.clamp(torch.abs(arr) - tau, min=0)

        return a
            
    def check_set_dims(self, Y, D):    
        # force Y to 3D and D to 2D.  Error if larger.
        if Y.ndim == 2:
            Y = rearrange(Y, 'h w -> 1 h w')
        elif Y.ndim == 1:
            Y = rearrange(Y, 'h -> 1 h 1')
        elif Y.ndim > 3:
            raise ValueError(f'Y of dim {Y.ndim} is not compatible')
        
        if D.ndim == 1:
            D = rearrange(D, 'w -> 1 w')
        elif D.ndim > 2:
            raise ValueError(f'Dictionary must be 2 dimensional not {D.ndim}')
            
        self.batch_size, self.Ty, self.joints = Y.shape    
        self.T, self.Np = D.shape
        
        assert self.Ty == self.T, 'Y dimensions must match the dictionary dimension'
         # print(f'{batch_size}, {T}, {Np}, {joints}')

class Dyan(BaseConfig):
    def __init__(self, device, config_filename=None, train_new_dict=None):
        super().__init__(config_filename)
        self.device = device
        
        self.train_new_dict = train_new_dict
        # self.dictionary_ = torch.zeros([self.T, self.Np]).to(self.device)
        # self.register_buffer('dictionary_', torch.zeros([self.T, self.Np]), persistent=False)
        # self.register_buffer('one', torch.Tensor([1.0]), persistent=False)
        # self.register_buffer('r_pow', torch.zeros(1), persistent=False)
               
        if self.dictionary_filename is not None:
            self.dictionary_path = Path().cwd().joinpath('data', self.dictionary_filename)
            self.load_raw_dict()
            self.generate_dictionary()
            print(f'Loaded pre-trained dictionary poles from {self.dictionary_path}', end='')
        elif self.train_new_dict == True and self.dictionary_filename == None:
            self.radius, self.angles = get_Drr_Dtheta(self.Np)
            self.r = nn.Parameter(torch.Tensor(self.radius), requires_grad=True)
            self.theta = nn.Parameter(self.angles, requires_grad=True)
        else:
            raise AttributeError
        
        # print(f'pow {r_pow.device} dic {self.dictionary_.device} one {self.one.device} r {self.r.device} th {self.theta.device}')
        
        self.fista = FISTA(self.device, self.fista_lambd, 
                           self.fista_reweighted, self.fista_iter, self.fista_tol, self.eps, self.normalize_dict)

    # def load_raw_dict(self):
    #     self.raw_data = torch.load(self.dictionary_path, map_location=self.device)['state_dict']
    #     self.rad = self.raw_data['sparseCoding.rr']
    #     self.theta_ = self.raw_data['sparseCoding.theta']
        

    def generate_random_poles(self):
        # uniform random radii within epsilon
        self.N = int((self.Np - 1) // 2)
        epsilon = 0.15
        r_min = (1 - epsilon)
        r_max = (1 + epsilon)
        self.radius = self.uniform_random_interval(r_min, r_max, self.N)
        # print(self.r.shape)

        # uniform random angles divided equally between quadrants
        polar_plot = torch.arange(4)
        quad_width = torch.pi / 2
        quad_angles = []
        poles_per_q = int(self.N // len(polar_plot))
        for quadrants in polar_plot:
            start = quadrants * quad_width
            end = (quadrants + 1) * quad_width
            quad_angles.append(self.uniform_random_interval(start, end, poles_per_q))
        
        self.angles = torch.cat(quad_angles)
        # print(self.theta.shape)
        
    def uniform_random_interval(self, r1, r2, size):
        return (r1 - r2) * torch.rand(size) + r2
        
    def normalize_matrix(self, arr):
        self.column_norm = torch.linalg.vector_norm(arr, dim=0, keepdims=True)
        return arr / (self.column_norm) # prevent divide by zero
        
    def generate_dictionary(self):
        poles = []
        
        for i in range(0,self.T):
            self.one = torch.Tensor([1]).to(self.device)
            self.r_pow = torch.pow(self.r, i).to(self.device)
            poles.append(torch.cat((self.one, 
                                    self.r_pow * torch.cos(i*self.theta),
                                    self.r_pow * torch.sin(i*self.theta)), axis=0))
            
        self.dictionary_ = torch.stack((poles), 0)
        # print(f'dict shape {self.dictionary_.shape}')
        
        if self.normalize_dict:
            self.dictionary_config = self.normalize_matrix(self.dictionary_)
            # print(' and normalized it.')
        else:
            self.dictionary_config = self.dictionary_
            print('. The dictionary is NOT normalized.\n\n')
    
    def dictionary(self, not_normalized=None):
        if not_normalized not in (None, True, False):
            raise ValueError(f'kwarg not_normalized must be None, True or False')
    
        if not_normalized is True:
            return self.dictionary_
        elif not_normalized is False:
            return self.normalize_matrix(self.dictionary_)
        else:
            return self.dictionary_config
    
    def dictionary_norm(self):
        return self.normalize_matrix(self.dictionary_)
    
    def forward(self, y):
        # self.display_attributes('device')
        # print(f'r {self.r.device} th {self.theta.device}')
        self.generate_dictionary()
        # print(f'r {self.r.device} th {self.theta.device}')
        C_pred = self.fista.solve_fista(y, self.dictionary_config)
        Y_pred = torch.matmul(self.dictionary_, C_pred)
        return Y_pred, C_pred
    
    def display_attributes(self, attribute_name):
        for name, value in self.__dict__.items():
            if hasattr(value, attribute_name):
                # Access and display the attribute
                print(f"{name}: {getattr(value, attribute_name)}")