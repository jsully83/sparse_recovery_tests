
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange
import yaml

class BaseConfig(nn.Module):
    def __init__(self, config_filename=None):
        super().__init__()
        
        self.required_keys = ['dictionary_filename', 'batch_size', 'T', 'Np', 'normalize_dict',
                              'fista_iter', 'fista_lambd', 'fista_reweighted', 'fista_tol']
        
        self.dictionary_filename = None
        self.batch_size = None
        self.T = None
        self.Np = None
        self.normalize_dict = None
        self.fista_iter = None
        self.fista_lambd = None
        self.fista_reweighted = None
        self.fista_tol = None
        
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
            
            print(f"Loaded configuration from {config_path}")

class FISTA(BaseConfig):
    def __init__(self, config_filename):
        super().__init__(config_filename)
        self.lambda_init = self.fista_lambd
        self.eps = 1e-10
        
    def solve_fista(self, Y, D):
        self.check_set_dims(Y, D)

        # create placeholders for the sparse vector
        self.C_curr = torch.zeros([self.batch_size, self.Np, self.joints])
        self.C_prev = self.C_curr.clone()
        self.t_prev = torch.Tensor([1.0])
        
        # precompute some mats
        Dt = D.transpose(0,1)
        self.DtD = torch.matmul(Dt, D) # [Np,Np]
        self.DtY = torch.matmul(Dt, Y)# [Batch, Np, joints]
        self.L = 1.0 / torch.linalg.matrix_norm(self.DtD, ord=2)
        
        # self.A = torch.eye(self.DtD.shape[0]) - self.L * self.DtD
        # self.B = self.L * self.DtY
        self.lambda_ = self.lambda_init
        if self.reweighted:
            for i in range(2):
                C = self.converge()
                self.weight_lambda()
        else:
            C = self.converge()
    
        return C
    
    def converge(self):
        for i in range(self.fista_iter):
            # calculate the gradient and soft thresholding step
            gradient = torch.matmul(self.DtD, self.C_prev) - self.DtY # [Np, Np] * [B, Np, joints] = [B, Np, joints] - [B, Np, joints]
            descent = self.C_prev - self.L*gradient # [B, Np, joints] - L*[B, Np, joints]
            C_next = self.soft_thresholding(descent, self.L*self.lambda_)
        
            # exit if it's close
            if torch.linalg.norm(C_next - self.C_curr) < self.tol:
                break

            # find the updated momentum constant
            t_next = (1 + (1 + 4 * self.t_prev**2)**0.5) / 2
            t_const = (self.t_prev - 1) / t_next 
            self.t_prev.copy_(t_next)
            
            # momentum step
            self.C_prev.copy_(C_next + t_const * (C_next - self.C_curr))

            self.C_curr.copy_(C_next)

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

class Dyan(FISTA):
    def __init__(self, device, config_filename=None):
        super().__init__(config_filename)

        self.device = device
        self.dictionary_ = torch.zeros([self.T, self.Np]).to(self.device)
        self.one = torch.Tensor([1.0 + self.eps]).to(self.device) # prevent div by 0
        
        if self.dictionary_filename is not None:
            self.dictionary_path = Path().cwd().joinpath('data', self.dictionary_filename)
            self.load_raw_dict()
            print(f'Loaded pre-trained dictionary poles from {self.dictionary_path}', end='')
        else:
                    # uniform random radii within epsilon
            self.N = int((self.Np - 1) // 2)
            epsilon = 0.15
            r_min = (1 - epsilon)
            r_max = (1 + epsilon)
            radii = torch.Tensor(self.uniform_random_interval(r_min, r_max, self.N))
            self.r = nn.Parameter(radii, requires_grad=True).to(self.device)
            # print(self.r.shape)

            # uniform random angles divided equally between quadrants
            polar_plot = torch.arange(4)
            quad_width = torch.pi / 2
            angles = []
            poles_per_q = int(self.N // len(polar_plot))
            for quadrants in polar_plot:
                start = quadrants * quad_width
                end = (quadrants + 1) * quad_width
                angles.append(self.uniform_random_interval(start, end, poles_per_q))
            
            theta_ = torch.Tensor(torch.cat(angles))
            self.theta = nn.Parameter(theta_, requires_grad=True).to(self.device)
            
            # self.generate_random_poles()
            print(f'Generated random dictionary poles', end='')
            print(f'r {self.r.device} th {self.theta.device}')
                
        self.generate_dictionary()
        # print(f'pow {r_pow.device} dic {self.dictionary_.device} one {self.one.device} r {self.r.device} th {self.theta.device}')


    def load_raw_dict(self):
        self.raw_data = torch.load(self.dictionary_path, map_location=self.device)['state_dict']
        self.r = self.raw_data['sparseCoding.rr']
        self.theta = self.raw_data['sparseCoding.theta']

    def generate_random_poles(self):
        # uniform random radii within epsilon
        self.N = int((self.Np - 1) // 2)
        epsilon = 0.15
        r_min = (1 - epsilon)
        r_max = (1 + epsilon)
        self.r = nn.Parameter(self.uniform_random_interval(r_min, r_max, self.N)).to(self.device)
        # print(self.r.shape)

        # uniform random angles divided equally between quadrants
        polar_plot = torch.arange(4)
        quad_width = torch.pi / 2
        angles = []
        poles_per_q = int(self.N // len(polar_plot))
        for quadrants in polar_plot:
            start = quadrants * quad_width
            end = (quadrants + 1) * quad_width
            angles.append(self.uniform_random_interval(start, end, poles_per_q))
        
        self.theta = nn.Parameter(torch.cat(angles)).to(self.device)
        # print(self.theta.shape)
        
    def uniform_random_interval(self, r1, r2, size):
        return (r1 - r2) * torch.rand(size) + r2
        
    def normalize_matrix(self, arr):
        self.column_norm = torch.linalg.vector_norm(arr, dim=0, keepdims=True)
        return arr / (self.column_norm) # prevent divide by zero
        
    def generate_dictionary(self):
        # print(f'r {self.r.device} th {self.theta.device}')
        # self.one = self.one.to(self.device)
        # self.r = self.r.to(self.device)
        # self.theta.to(self.device)
        # self.dictionary_ = self.dictionary_.to(self.device)
        
        for i in range(0,self.T):
            # self.r_pow = torch.pow(self.r,i).to(self.device)
            self.r_pow = torch.pow(self.r,i)
            # print(f'pow {self.r_pow.device} dic {self.dictionary_.device} one {self.one.device} r {self.r.device} th {self.theta.device}')
            self.dictionary_[i,:] = torch.cat((self.one, 
                                               self.r_pow * torch.cos(i*self.theta),
                                               self.r_pow * torch.sin(i*self.theta)), axis=0)
            
        
        if self.normalize_dict:
            self.dictionary_config = self.normalize_matrix(self.dictionary_)
            print(' and normalized it.')
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
        print(f'r {self.r.device} th {self.theta.device}')
        self.generate_dictionary()
        print(f'r {self.r.device} th {self.theta.device}')
        C_pred = self.fista.solve(y, self.dictionary_config)
        Y_pred = torch.matmul(self.dictionary_, C_pred)
        return Y_pred, C_pred
    
   