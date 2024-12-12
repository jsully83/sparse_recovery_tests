import torch
import torch.nn as nn
from einops import rearrange

class FISTA(nn.Module):
    def __init__(self, n_iters, lambda_, tol, reweighted=True):
        super().__init__()
        self.n_iters = n_iters
        self.lambda_ = torch.Tensor([lambda_]).cuda()  # Initial lambda
        self.tol = tol
        self.reweighted = reweighted
        self.eps = 1e-10
    
    def solve(self, Y, D):
        """
        Solve for the sparse vector C in Y = D * C with FISTA or Reweighted FISTA.
        """
        self.check_set_dims(Y, D)

        # Initialize sparse code placeholders
        self.C_curr = torch.zeros(self.batch_size, self.Np, self.joints).cuda()
        self.C_prev = self.C_curr.clone()
        self.t_prev = torch.Tensor([1]).cuda()
        
        # Precompute matrix products and Lipschitz constant
        Dt = D.transpose(0, 1)
        self.DtD = torch.matmul(Dt, D) 
        self.DtY = torch.matmul(Dt, Y)
        self.L = 1 / torch.linalg.matrix_norm(self.DtD, ord=2)

        if self.reweighted:
            # Initialize lambda_ as a tensor for element-wise thresholding
            self.lambda_ = self.lambda_ * torch.ones(self.batch_size, self.Np, self.joints).cuda()
            
            for _ in range(2): 
                self.converge()        # Run FISTA with current lambda_
                self.weight_lambda()   # Update lambda_ based on the latest C_curr
                print(f"C_curr sum after reweighting: {self.C_curr.sum()}")
            return self.C_curr
            
        else:
            # Run standard FISTA without reweighting
            return self.converge()
    
    def converge(self):
        """
        Run FISTA iterations with gradient descent and reweighted soft-thresholding.
        """
        for i in range(self.n_iters):
            # Gradient descent step
            gradient = torch.matmul(self.DtD, self.C_prev) - self.DtY
            descent = self.C_prev - self.L * gradient
            
            # Apply soft-thresholding with the current (reweighted) lambda
            C_next = self.soft_thresholding(descent, self.L * self.lambda_)
        
            # Check for convergence
            if torch.linalg.norm(C_next - self.C_curr) < self.tol:
                break

            # Update the momentum parameter
            t_next = (1 + (1 + 4 * self.t_prev**2)**0.5) / 2
            t_const = (self.t_prev - 1) / t_next 
            self.t_prev.copy_(t_next)
            
            # Momentum step
            self.C_prev.copy_(C_next + t_const * (C_next - self.C_curr))
            self.C_curr.copy_(C_next)
            
            print(f'Iteration {i}: norm diff = {torch.linalg.norm(C_next - self.C_curr)}, C_curr sum = {self.C_curr.sum().item()}')
        
        return self.C_curr
    
    def weight_lambda(self):
        """
        Update weights for reweighted FISTA.
        """
        # Update lambda_ by applying reweighting based on the current C_curr
        self.weights = 1 / (torch.abs(self.C_curr) + self.eps)
        # Normalize weights to maintain stability
        self.weights = self.weights / torch.linalg.vector_norm(self.weights, dim=1, keepdims=True)
        self.lambda_ = self.lambda_ * self.weights * self.Np  # Reweighted lambda update
                    
    def soft_thresholding(self, arr, tau):
        """
        Element-wise soft-thresholding with adaptive threshold tau.
        """
        return torch.sign(arr) * torch.clamp(torch.abs(arr) - tau, min=0)
            
    def check_set_dims(self, Y, D):    
        """
        Force Y and D to required dimensions and ensure compatibility.
        """
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