############################# Import Section #################################
import sys
sys.path.append('../')
sys.path.append('../data')
sys.path.append('.')

import ipdb
from math import sqrt
import numpy as np

import torch
import torch.nn as nn

from utils import random
# from modelZoo.actHeat import imageFeatureExtractor

random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
############################# Import Section #################################


def creatRealDictionary(T, rr, theta, gpu_id):
    WVar = []
    Wones = torch.ones(1).cuda(gpu_id)
    Wones  = Wones
    for i in range(0,T):
        W1 = torch.mul(torch.pow(rr,i) , torch.cos(i * theta))
        W2 = torch.mul (torch.pow(rr,i) , torch.sin(i *theta) )
        W = torch.cat((Wones,W1,W2),0)
        WVar.append(W.view(1,-1))
    dic = torch.cat((WVar),0)

    return dic


def fista_new(D, Y, lambd, maxIter, gpu_id):
    """
        D: (T, num_pole)
        Y: ((batch_size x num_clips), T, num_joints x dim_joints)
    """
    DtD = torch.matmul(torch.t(D),D)
    L = torch.norm(DtD,2)
    linv = 1/L
    DtY = torch.matmul(torch.t(D),Y)
    x_old = torch.zeros(DtD.shape[1],DtY.shape[2]).cuda(gpu_id)
    t = 1
    y_old = x_old
    lambd = lambd*(linv.data.cpu().numpy())
    # print('lambda:', lambd, 'linv:',1/L, 'DtD:',DtD, 'L', L )
    # print('dictionary:', D)
    A = torch.eye(DtD.shape[1]).cuda(gpu_id) - torch.mul(DtD,linv)
    DtY = torch.mul(DtY,linv)
    Softshrink = nn.Softshrink(lambd)
    for ii in range(maxIter):
        # print('iter:',ii, lambd)
        Ay = torch.matmul(A,y_old)
        del y_old
        x_new = Softshrink((Ay + DtY)+1e-6)

        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2.
        tt = (t-1)/t_new
        y_old = torch.mul( x_new,(1 + tt))
        y_old -= torch.mul(x_old , tt)
        # pdb.set_trace()
        if torch.norm((x_old - x_new),p=2)/x_old.shape[1] < 1e-5:
            x_old = x_new
            # print('Iter:', ii)
            break
        t = t_new
        x_old = x_new
        del x_new
    
    return x_old


def fista_reweighted(D, Y, lambd, w, maxIter):
    '''
        D: [T, 161]\\
        Y: [Nsample, T, 50]\\
        w: [Nsample, 161, 25 x 2]
    '''
    if len(D.shape) < 3:
        DtD = torch.matmul(torch.t(D), D)
        DtY = torch.matmul(torch.t(D), Y)
    else:
        DtD = torch.matmul(D.permute(0, 2, 1), D)
        DtY = torch.matmul(D.permute(0, 2, 1), Y)
    L = torch.norm(DtD, 2) #spectral norm/largest singular value of DtD
    Linv = 1/L
    weightedLambd = (w*lambd) * Linv.data.item()
    x_old = torch.zeros(DtD.shape[1], DtY.shape[2]).to(D.device)
    # x_old = x_init
    y_old = x_old
    A = torch.eye(DtD.shape[1]).to(D.device) - torch.mul(DtD,Linv)
    const_xminus = torch.mul(DtY, Linv) - weightedLambd.to(D.device)
    const_xplus = torch.mul(DtY, Linv) + weightedLambd.to(D.device)

    t_old = 1
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


class DyanEncoder(nn.Module):
    def __init__(self, Drr, Dtheta, T, 
                 wiRH,
                 lam,
                 gpu_id, freezeD=False):
        super(DyanEncoder, self).__init__()
        self.rr = nn.Parameter(Drr)
        self.theta = nn.Parameter(Dtheta)
        self.T = T
        self.wiRH = wiRH
        self.lam = lam
        self.gpu_id = gpu_id
        self.freezeD = freezeD
        # Frozen Dictionary, no need to create each forward
        if freezeD:  self.creat_D()

    def creat_D(self):
        """Create the Dictionary if the rho and theta are fixed
        """
        self.D = creatRealDictionary(self.T, self.rr, self.theta, self.gpu_id)

    def RH_FISTA(self, x):
        '''With Reweighted Heuristic Algorithm
        x: N x T x (num_j x dim_j)
        '''
        if self.freezeD:      D = self.D.detach() # Detach to ensure it's not part of a computation graph
        else:                 
            D = creatRealDictionary(self.T, self.rr, self.theta, self.gpu_id)
            D_norm = D / torch.linalg.vector_norm(D, dim=0, keepdim=True)
        # ipdb.set_trace()
        # batch_size, num_poles, num_joints x dim_joint
        Nsample, Npole, Ddata = x.shape[0], D.shape[1], x.shape[2]
        w_init = torch.ones(Nsample, Npole, Ddata)

        i = 0
        while i < 2:
            # temp: [N, Np, num_j x dim_j]
            temp = fista_reweighted(D_norm, x, self.lam, w_init, 100)
            # temp = fista_reweighted(D, x, self.lam, w_init, 100)
            'for vector:'
            w = 1 / (torch.abs(temp) + 1e-6)
            # Scaler for Reweight
            # Batch-wise
            # w_init = (w/torch.norm(w)) * D.shape[1]
            # Matrix-wise
            # w_init = (w/torch.norm(w, dim=(1,2)).view(-1,1,1)) * Npole
            # Column-wise for Coefficients
            w_init = (w/torch.linalg.vector_norm(w, ord=1, dim=1, keepdim=True)) * Npole
            # print(torch.linalg.vector_norm(w, ord=1, dim=1, keepdim=True).shape)
            # w_init = (w/torch.norm(w, dim=(1)).view(Nsample,1,-1)) * Npole
            
            final = temp
            del temp
            i += 1

        sparseCode = final
        
        reconst = torch.matmul(D, sparseCode.cuda(self.gpu_id))

        return sparseCode, D, reconst

    def FISTA(self,x):
        """Without Reweighted Heuristic Algorithm
        """
        if self.freezeD:  dic = self.D
        else:             dic = creatRealDictionary(self.T, self.rr, self.theta, self.gpu_id)
        # FISTA
        sparseCode = fista_new(dic, x, self.lam, 100, self.gpu_id)
        # # FISTA loop
        # bs = x.shape[0]
        # list_C = []
        # for i_b in range(bs):
        #     list_C.append(fista_new(dic, x[i_b:i_b+1], self.lam, 100, self.gpu_id))
        # sparseCode = torch.cat(list_C,dim=0)
        
        reconst = torch.matmul(dic, sparseCode)
        # return sparseCode, dic, reconst
        return sparseCode, dic, reconst
    
    def forward(self, x):
        if self.wiRH:   return self.RH_FISTA(x)
        else:           return self.FISTA(x)
