import torch
import torch.nn as nn
from torch.nn.functional import softshrink
from einops import rearrange

"""
Class that implements FISTA
"""
class FISTA(nn.Module):
    def __init__(self, n_iters, lambda_, tol, reweighted=True):
        super().__init__()
        self.shrink = nn.Softshrink(lambda_)
        self.n_iters = n_iters
        self.lambda_init = lambda_
    
        self.tol = tol
        self.reweighted = reweighted
        self.eps = 1e-10
        
    def solve(self, Y, D):
        self.check_set_dims(Y, D)

        # create placeholders for the sparse vector
        self.C_curr = torch.zeros([self.batch_size, self.Np, self.joints]).cuda()
        self.C_prev = self.C_curr.clone()
        self.t_prev = torch.Tensor([1.0]).cuda()
        
        # precompute some mats
        Dt = D.transpose(0,1)
        self.DtD = torch.matmul(Dt, D) # [Np,Np]
        self.DtY = torch.matmul(Dt, Y)# [Batch, Np, joints]
        self.L = 1.0 / torch.linalg.matrix_norm(self.DtD, ord=2)
        
        # self.A = torch.eye(self.DtD.shape[0]).cuda() - self.L * self.DtD
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
        for i in range(self.n_iters):
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