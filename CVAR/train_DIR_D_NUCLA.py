# DIR-1: train dictionary, N-UCLA dataset
from ast import parse
import os
import ipdb
import time
import numpy as np
from einops import rearrange
import random
import argparse
import datetime
# from ptflops import get_model_complexity_info
from matplotlib import pyplot as plt
import wandb

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

# from dataset.crossView_UCLA import NUCLA_CrossView
from dataset.crossView_UCLA_ske import NUCLA_CrossView
from modelZoo.BinaryCoding import DyanEncoder, binarizeSparseCode
from utils import sparsity, gridRing

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

torch.set_printoptions(precision=10)


def get_parser():
    def str2bool(v):
        if    v.lower() in ('yes', 'true', 't', 'y', '1'):  return True
        elif  v.lower() in ('no', 'false', 'f', 'n', '0'):  return False
        else:  raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
    parser = argparse.ArgumentParser(description='CVARDIF')
    parser.add_argument('--modelRoot', default='exps_should_be_saved_on_HDD',
                        help='the work folder for storing experiment results')
    parser.add_argument('--path_list', default='', help='')
    parser.add_argument('--cus_n', default='', help='customized name')
    parser.add_argument('--mode', default='dy+bi')
    parser.add_argument('--wiRH', default='1', type=str2bool, help='Use Reweighted Heuristic Algorithm')
    parser.add_argument('--setup', default='setup1', help='')
    parser.add_argument('--dataType', default='2D', help='')
    parser.add_argument('--sampling', default='Single', help='')
    parser.add_argument('--nClip', default=1, type=int, help='') # sampling=='multi' or sampling!='Single'

    parser.add_argument('--T', default=36, type=int, help='')
    parser.add_argument('--N', default=80*2, type=int, help='')
    parser.add_argument('--lam_f', default=0.1, type=float)
    parser.add_argument('--gumbel_thresh', default=0.505, type=float) # 0.501/0.503/0.510

    parser.add_argument('--gpu_id', default=0, type=int, help='')
    # parser.add_argument('--gpu_id', default='cuda:3', type=str, help='')
    parser.add_argument('--bs', default=8, type=int, help='')
    parser.add_argument('--nw', default=8, type=int, help='')
    parser.add_argument('--Epoch', default=100, type=int, help='')
    parser.add_argument('--Alpha', default=5e-1, type=float, help='bi loss')
    parser.add_argument('--lam2', default=1, type=float, help='mse loss')
    parser.add_argument('--lr_2', default=1e-3, type=float)
    parser.add_argument('--run_name', default='test_name', type=str)

    return parser

def main(args):
    os.system('date')
    # Configurations
    ## Paths
    args.bs_t = 64
    str_conf = f"{'wiRH' if args.wiRH else 'woRH'}"
    print(f" {args.mode} | {str_conf} | Batch Size: Train {args.bs} | Test {args.bs_t} ")
    print(f"\tlam_f: {args.lam_f} | Alpha: {args.Alpha} | lam2: {args.lam2} | lr_2: {args.lr_2} | g_t: {args.gumbel_thresh}")
    args.saveModel = os.path.join(args.modelRoot,
                                  f"NUCLA_CV_{args.setup}_{args.sampling}/DIR_D_{str_conf}/")
    if not os.path.exists(args.saveModel): os.makedirs(args.saveModel)
    ## Dictionary
    P, Pall = gridRing(args.N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()
    ## Network
    # net = binarizeSparseCode(Drr, Dtheta, args.T, args.wiRH,
    #                          args.gpu_id, Inference=False, 
    #                          fistaLam=args.lam_f)
    
    net = DyanEncoder(Drr, Dtheta, args.T, args.wiRH, args.lam_f, args.gpu_id)
    
    net.cuda(args.gpu_id)
    # Dataset
    assert args.path_list!='', '!!! NO Dataset Samples LIST !!!'
    path_list = args.path_list + f"data/CV/{args.setup}/"
    trainSet = NUCLA_CrossView(root_list=path_list, phase='train',
                               setup=args.setup, dataType=args.dataType,
                               sampling=args.sampling, nClip=args.nClip,
                               T=args.T, maskType='None')
    trainloader = DataLoader(trainSet, shuffle=True,
                             batch_size=args.bs, num_workers=args.nw)
    testSet = NUCLA_CrossView(root_list=path_list, phase='test',
                              setup=args.setup, dataType=args.dataType,
                              sampling=args.sampling, nClip=args.nClip,
                              T=args.T, maskType='None')
    testloader = DataLoader(testSet, shuffle=False,
                            batch_size=args.bs_t, num_workers=args.nw)
    
    # train_views = [sample[0] for sample in trainSet.samples_list]
    # print(np.unique(train_views, return_counts=True))
    
    # test_views = [sample for sample in testSet.samples_list]
    # print(np.unique(test_views))
    
    # Training Strategy
    optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), 
                                lr=args.lr_2, weight_decay=0.001, momentum=0.9)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)
    # Loss
    mseLoss = torch.nn.MSELoss()
    l1Loss = torch.nn.L1Loss(reduction='mean')

    # Loss = []
    for epoch in range(0, args.Epoch+1):
        print('training epoch:', epoch)
        net.train()

        lossVal,lossMSE, lossL1, L1sparse = [], [], [], []
        mseB = []
        start_time = time.time()
        for _, sample in enumerate(trainloader):
            # print('sample:', i)
            optimizer.zero_grad()
            # batch_size, num_clips, t, num_joint, dim_joint
            skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(args.gpu_id)
            # -> batch_size x num_clips, t, num_joint x dim_joint
            input_skeletons = rearrange(skeletons, 'n c t j d -> (n c) t (j d)')
            ### rh-dyan + bi
            # binaryCode, output_skeletons, _, R_B = net(input_skeletons,
            #                                            args.gumbel_thresh,
            #                                            False)
            
            sparse_code, D, output_skeletons = net(input_skeletons)
                    
            # target_coeff = torch.zeros_like(binaryCode).cuda(args.gpu_id)
            l1 = l1Loss(output_skeletons, input_skeletons)
            l1_sparse = (torch.linalg.vector_norm(sparse_code, ord=1, dim=1, keepdim=True) * args.lam_f).mean()
            loss = mseLoss(output_skeletons, input_skeletons) + l1_sparse
            # loss = args.lam2*mseLoss(output_skeletons, input_skeletons)
            
            # loss = args.lam2*mseLoss(output_skeletons, input_skeletons
            #         ) + args.Alpha*l1Loss(binaryCode,target_coeff)
            
            #### BP and Log
            loss.backward()
            optimizer.step()
            # ipdb.set_trace()
            lossMSE.append((mseLoss(output_skeletons, input_skeletons) + l1_sparse).detach().item())
            lossL1.append((l1Loss(output_skeletons, input_skeletons)).detach().item())
            L1sparse.append(l1_sparse.detach().item())
            # mseB.append((args.lam2*mseLoss(R_B, input_skeletons)).detach().item())
            lossVal.append(loss.detach().item())

        end_time = time.time()
        
        train_loss_L1 = np.mean(np.asarray(lossL1))
        train_loss_mse = np.mean(np.asarray(lossMSE))
        train_L1_sparse = np.mean(np.asarray(L1sparse))
        # train_loss_mseB = np.mean(np.asarray(mseB))
        
        
        print('epoch:', epoch, 'loss:', np.mean(np.asarray(lossVal)), 'time(h):', (end_time - start_time) / 3600)
        print('Train epoch:', epoch, 'mse loss:', train_loss_mse,
            #   'mseB loss:', train_loss_mseB,
              'L1 loss:', train_loss_L1,
              f'duration:{(end_time - start_time) / 60:.4f} min')

        if epoch % 1 == 0:
            torch.save({'epoch(Train)': epoch + 1, 'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict()}, args.saveModel +'dir_d_'+ str(epoch) + '.pth')
            net.eval()
            with torch.no_grad():

                ERROR,ERROR_B = [], []
                l1 = []
                l1_loss = []
                Sp_0, Sp_th = [],[]
                for _, sample in enumerate(testloader):
                    # batch_size, num_clips, t, num_joint, dim_joint
                    skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(args.gpu_id)
                    # -> batch_size x num_clips, t, num_joint x dim_joint
                    input_skeletons = rearrange(skeletons, 'n c t j d -> (n c) t (j d)')
                    # C, output_skeletons, _, R_B = net(input_skeletons,
                    #                                 args.gumbel_thresh,
                    #                                 True)
                    # print(sample['cam'])
                    sparse_code_test, D, output_skeletons = net(input_skeletons)
                    
                    l1_sparse = (torch.linalg.vector_norm(sparse_code_test, ord=1, dim=1, keepdim=True) * args.lam_f).mean()
                    error   = mseLoss(output_skeletons, input_skeletons).cpu() + l1_sparse
                    # error_b = args.lam2*mseLoss(R_B, input_skeletons).cpu()
                    ERROR.append(error.detach().item())
                    l1.append(l1_sparse.detach().item())
                    l1_loss.append(l1Loss(output_skeletons, input_skeletons).detach().item())
                    # ERROR_B.append(error_b) 
                    # sp_0,sp_th =sparsity(C)  
                    # Sp_0.append(sp_0)
                    # Sp_th.append(sp_th)

                    test_loss_mse = np.mean(np.asarray(ERROR))
                    test_L1sparse = np.mean(np.asarray(l1))
                    test_l1_loss = np.mean(np.asarray(l1_loss))

                print('Test epoch:', epoch, f'MSE_Y_C:{test_loss_mse}')
                                            #  f'MSE_Y_B:{torch.mean(torch.tensor(ERROR_B))}')
                # print(f'\tSp_0:{torch.mean(torch.tensor(Sp_0))}', 
                #       f'Sp_th:{torch.mean(torch.tensor(Sp_th))}')
        
        wandb.log({'train_loss_mse+L1':train_loss_mse,
                   'train_L1sparse':train_L1_sparse,
                   'train_L1_loss':train_loss_L1,
                   'test_loss_mse+L1':test_loss_mse,
                   'test_L1sparse':test_L1sparse,
                   'test_L1_loss':test_l1_loss
                   })
        
        scheduler.step()
    print('done')

if __name__ == "__main__":
    
    parser = get_parser()
    args=parser.parse_args()
    
    wandb.init(
        # set the wandb project where this run will be logged
        project="CVAR",
        name = args.run_name,

        # track hyperparameters and run metadata
        config={
        "Lam_f": args.lam_f,
        "architecture": "DYAN",
        "dataset": "N-UCLA",
        "epochs": 100,
        })   
    
    main(args)
