# DIR-1: train dictionary, N-UCLA dataset
from ast import parse
import os
import ipdb
import time
import numpy as np
import random
import argparse
import datetime
# from ptflops import get_model_complexity_info
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

# from dataset.crossView_UCLA import NUCLA_CrossView
from dataset.crossView_UCLA_ske import NUCLA_CrossView
from modelZoo.BinaryCoding import DyanEncoder, binarizeSparseCode
from utils import gridRing
from torch.utils.tensorboard import SummaryWriter

seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

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
    parser.add_argument('--setup', default='setup1', help='')
    parser.add_argument('--dataType', default='2D', help='')
    parser.add_argument('--sampling', default='Single', help='')
    parser.add_argument('--nClip', default=1, type=int, help='') # sampling=='multi' or sampling!='Single'

    parser.add_argument('--T', default=36, type=int, help='')
    parser.add_argument('--N', default=80*2, type=int, help='')
    parser.add_argument('--lam_f', default=0.1, type=float)
    parser.add_argument('--gumbel_thresh', default=0.505, type=float) # 0.501/0.503/0.510

    parser.add_argument('--gpu_id', default=0, type=int, help='')
    parser.add_argument('--bs', default=8, type=int, help='')
    parser.add_argument('--nw', default=8, type=int, help='')
    parser.add_argument('--Epoch', default=100, type=int, help='')
    parser.add_argument('--Alpha', default=1e-1, type=float, help='bi loss')
    parser.add_argument('--lam2', default=5e-1, type=float, help='mse loss')
    parser.add_argument('--lr_2', default=1e-4, type=float)

    return parser

def main(args):
    # Configurations
    ## Paths
    args.saveModel = os.path.join(args.modelRoot,
                                  f'NUCLA_CV_{args.setup}_{args.sampling}/DIR_D_{args.mode}/')
    if not os.path.exists(args.saveModel): os.makedirs(args.saveModel)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(args.saveModel)
    ## Dictionary
    P, Pall = gridRing(args.N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()
    ## Network
    net = binarizeSparseCode(num_binary=128, Drr=Drr, Dtheta=Dtheta,
                             gpu_id=args.gpu_id, Inference=False, 
                             fistaLam=args.lam_f)
    net.cuda(args.gpu_id)
    # Dataset
    assert args.path_list!='', '!!! NO Dataset Samples LIST !!!'
    path_list = args.path_list + f"/data/CV/{args.setup}/"
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
                            batch_size=args.bs, num_workers=args.nw)
    # Training Strategy
    optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()), 
                                lr=args.lr_2, weight_decay=0.001, momentum=0.9)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)
    # Loss
    mseLoss = torch.nn.MSELoss()
    l1Loss = torch.nn.L1Loss(reduction='mean')
    mse_test = torch.nn.MSELoss()

    views_train = [samples[0] for samples in trainSet.samples_list]
    print(np.unique(views_train, return_counts=True))
    # return

    net.train()
    # Loss = []
    for epoch in range(0, args.Epoch+1):
        print('training epoch:', epoch)
        lossVal = []
        lossMSE = []
        # lossL1 = []
        start_time = time.time()
        for i, sample in enumerate(trainloader):
            # print('sample:', i)
            optimizer.zero_grad()
            skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(args.gpu_id)
            t = skeletons.shape[1] # (batch_size x num_clips) x t x num_joint x dim_joint
            input_skeletons = skeletons.reshape(skeletons.shape[0], t, -1) # (batch_size x num_clips) x t x (dim_joint x num_joint)
            ### rh-dyan + bi
            binaryCode, output_skeletons, _ = net(input_skeletons, t, args.gumbel_thresh)
            # target_coeff = torch.zeros_like(binaryCode).cuda(args.gpu_id)
            loss = mseLoss(output_skeletons, input_skeletons)
            # loss = args.lam2*mseLoss(output_skeletons, input_skeletons) + args.Alpha*l1Loss(binaryCode,target_coeff)
            #### BP and Log
            loss.backward()
            optimizer.step()
            # ipdb.set_trace()
            lossMSE.append(mseLoss(output_skeletons, input_skeletons).data.item())
            # lossL1.append(args.Alpha*l1Loss(binaryCode,target_coeff).data.item())
            lossVal.append(loss.data.item())
        end_time = time.time()
        # print('epoch:', epoch, 'loss:', np.mean(np.asarray(lossVal)), 'time(h):', (end_time - start_time) / 3600)

        train_loss_mse = np.mean(np.asarray(lossMSE))
        # train_loss_L1 = np.mean(np.asarray(lossL1))
        print('epoch:', epoch, 'mse loss:', train_loss_mse,
            #   'L1 loss:', train_loss_L1,
              'duration:', (end_time - start_time) / 3600, 'hr')

        if epoch % 5 == 0:
            torch.save({'epoch': epoch + 1, 'state_dict': net.state_dict(),
                        'optimizer': optimizer.state_dict()}, args.saveModel +'dir_d_'+ str(epoch) + '.pth')
            testLossMSE = []
            with torch.no_grad():
                # ERROR = torch.zeros(testSet.__len__(), 1)

                for i, sample in enumerate(testloader):
                    skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(args.gpu_id)
                    t = skeletons.shape[1]
                    input_skeletons = skeletons.reshape(skeletons.shape[0], t, -1)
                    # 'regular dyan'
                    # _,_,output_skeletons = net.forward2(input_skeletons, t) # reconst
                    # output_skeletons = net.prediction(input_skeletons[:,0:t-1], t-1)

                    # 'rhDyan+Bi'
                    _, output_skeletons, _ = net(input_skeletons, t, args.gumbel_thresh)
                    # error = torch.norm(output_skeletons - input_skeletons).cpu()
                    # ERROR[i] = error
                    testLossMSE.append(mse_test(output_skeletons, input_skeletons).data.item())
                    # testLossMSE.append(args.lam2*mse_test(output_skeletons, input_skeletons).data.item())
                    
                    
                testloss_mse = np.mean(np.asarray(testLossMSE))
                # testing_error = torch.mean(ERROR)
                print('epoch:', epoch, 'error:', testloss_mse)
        
        scheduler.step()
        writer.add_scalar('Loss/Training Loss MSE', train_loss_mse, epoch)
        writer.add_scalar('Loss/Testing Loss MSE', testloss_mse, epoch)
        writer.add_scalars('Loss', {'Training Loss': train_loss_mse, 'Testing Loss': testloss_mse}, epoch)
           
    print('done')
    writer.close()

if __name__ == "__main__":
    parser = get_parser()
    args=parser.parse_args()
    main(args)
