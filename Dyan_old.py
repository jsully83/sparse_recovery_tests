import torch
import torch.nn as nn
from modelZoo.Fista import FISTA

import numpy as np

class Dyan(nn.Module):
    def __init__(self, device, d_path=None):
        super().__init__()
        
        self.tol = 0.01
        self.lambda_ = 0.005
        self.n_iters = 100
        
        self.fista = FISTA(self.n_iters, self.lambda_, self.tol, reweighted=True)
        
        self.device = device
        self.T = 36
        self.Np = 161
        self.eps = 1e-10
        self.d_path = d_path
        self.one = torch.Tensor([1.0 + self.eps]).cuda() # prevent div by 0
        
        if d_path is not None:            
            self.load_raw_dict()
            self.generate_dictionary()

    def load_raw_dict(self):
        self.raw_data = torch.load(self.d_path, map_location=self.device)['state_dict']
        self.r = self.raw_data['sparseCoding.rr']
        self.theta = self.raw_data['sparseCoding.theta']

    def normalize_matrix(self, arr):
        self.column_norm = torch.linalg.vector_norm(arr, dim=0, keepdims=True)
        return arr / (self.column_norm) # prevent divide by zero
        
    def generate_dictionary(self, norm=False):
        self.dictionary = torch.zeros([self.T, self.Np]).cuda()
        
        for i in range(0,self.T):
            r_pow = torch.pow(self.r,i)
            self.dictionary[i,:] = torch.cat((self.one , r_pow * torch.cos(i*self.theta),
                                     r_pow * torch.sin(i*self.theta)), axis=0)
        
        if norm:
            self.dictionary = self.normalize_matrix(self.dictionary)
            
    # def dictionary(self):
    #     return self.dictionary
    
    def dictionary_norm(self):
        return self.normalize_matrix(self.dictionary)
                 
    def forward(self, y):
        self.generate_dictionary(norm=True)
        C_pred = self.fista.solve(y, self.dictionary)
        Y_pred = torch.matmul(self.dictionary.double(), C_pred.double())
        return Y_pred, C_pred
    
    
    # def forward(self, y):
    #     self.generate_dictionary(norm=True)
    #     i=0
    #     w_init = torch.ones(y.shape[0], self.dictionary.shape[1], y.shape[2])
    #     while i < 2:
    #         temp = self.fista_reweighted(self.dictionary, y, self.lambda_, w_init, 100)
    #         'for vector:'
    #         w = 1 / (torch.abs(temp) + 1e-2)
    #         w_init = (w/torch.norm(w)) * self.dictionary.shape[1]
        
    #         C_pred = temp
    #         del temp
    #         i += 1
        
        
    #     Y_pred = torch.matmul(self.dictionary, C_pred)
    #     return Y_pred, C_pred

    def fista_reweighted(self, D, Y, lambd, w, maxIter):
        'D: T x 161, Y: N x T x 50, w: N x 161 x 50'
        if len(D.shape) < 3:
            DtD = torch.matmul(torch.t(D), D)
            DtY = torch.matmul(torch.t(D), Y)
            # print(f'D1: {D.shape}')
        else:
            DtD = torch.matmul(D.permute(0, 2, 1), D)
            DtY = torch.matmul(D.permute(0, 2, 1), Y)
            # print(f'D2: {D.shape}')
        # eig, v = torch.eig(DtD, eigenvectors=True)
        # eig, v = torch.linalg.eig(DtD)
        # L = torch.max(eig)
        # L = torch.norm(DtD, 2)
        L = torch.linalg.matrix_norm(DtD, ord=2)

        # eigs = torch.abs(torch.linalg.eigvals(DtD))
        # L = torch.max(eigs, dim=1).values

        Linv = 1/L
    
        weightedLambd = (w*lambd) * Linv.data.item()
        # print(f'w: {w.shape}')
        # print(f'Linv.data.item(): {Linv.data.item()}')
        # print(f'weightedLambd: {torch.unique(weightedLambd,return_counts=True)}')
        x_old = torch.zeros(DtD.shape[1], DtY.shape[2]).to(D.device)
        # x_old = x_init
        y_old = x_old
        A = torch.eye(DtD.shape[1]).to(D.device) - torch.mul(DtD,Linv)
        t_old = 1

        const_xminus = torch.mul(DtY, Linv) - weightedLambd.to(D.device)
        const_xplus = torch.mul(DtY, Linv) + weightedLambd.to(D.device)

        iter = 0

        while iter < maxIter:
            iter +=1
            Ay = torch.matmul(A, y_old)
            x_newminus = Ay + const_xminus
            x_newplus = Ay + const_xplus
            x_new = torch.max(torch.zeros_like(x_newminus), x_newminus) + \
                    torch.min(torch.zeros_like(x_newplus), x_newplus)

            t_new = (1 + np.sqrt(1 + 4 * t_old ** 2)) / 2.

            tt = (t_old-1)/t_new
            y_new = x_new + torch.mul(tt, (x_new-x_old))  # y_new = x_new + ((t_old-1)/t_new) *(x_new-x_old)
            if torch.norm((x_old - x_new), p=2) / x_old.shape[1] < 1e-5:
                x_old = x_new
                break

            t_old = t_new
            x_old = x_new
            y_old = y_new

        return x_old