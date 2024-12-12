# DIR-2: train CVAR, N-UCLA dataset
import time
import ipdb
import argparse
from matplotlib import pyplot as plt
# from ptflops import get_model_complexity_info

import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dataset.crossView_UCLA_ske import os, np, random, NUCLA_CrossView
from modelZoo.BinaryCoding import Fullclassification, nn, gridRing, classificationWSparseCode
from test_cls_CV_DIR import testing, getPlots
from utils import load_pretrainedModel_endtoEnd

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
    parser.add_argument('--pretrain',  default='', help='')
    parser.add_argument('--cus_n', default='', help='customized name')
    
    parser.add_argument('--dataset', default='NUCLA', help='')
    parser.add_argument('--setup', default='setup1', help='')
    parser.add_argument('--num_class', default=10, type=int, help='')
    parser.add_argument('--dataType', default='2D', help='')
    parser.add_argument('--sampling', default='Single', help='')
    parser.add_argument('--nClip', default=1, type=int, help='') # sampling=='multi' or sampling!='Single'
    parser.add_argument('--bs', default=8, type=int, help='')
    parser.add_argument('--nw', default=8, type=int, help='')
    
    parser.add_argument('--mode', default='dy+bi+cl', help='dy+bi+cl | dy+cl | rgb+dy')
    parser.add_argument('--RHdyan', default='1', type=str2bool, help='')
    parser.add_argument('--withMask', default='0', type=str2bool, help='')
    parser.add_argument('--maskType', default='None', help='')
    parser.add_argument('--fusion', default='0', type=str2bool, help='')
    parser.add_argument('--groupLasso', default='0', type=str2bool, help='')

    parser.add_argument('--T', default=36, type=int, help='')
    parser.add_argument('--N', default=80*2, type=int, help='')
    parser.add_argument('--lam_f', default=0.1, type=float)
    parser.add_argument('--gumbel_thresh', default=0.505, type=float) # 0.503

    parser.add_argument('--gpu_id', default=0, type=int, help='')
    parser.add_argument('--Epoch', default=100, type=int, help='')
    parser.add_argument('--lr', default=5e-4, type=float, help='sparse coding')
    parser.add_argument('--lr_2', default=1e-3, type=float, help='classifier')
    parser.add_argument('--Alpha', default=0.1, type=float, help='bi loss')
    parser.add_argument('--lam1', default=1, type=float, help='cls loss')
    parser.add_argument('--lam2', default=0.5, type=float, help='mse loss')

    return parser

def main(args):
    args.saveModel = os.path.join(args.modelRoot,
                                  f'NUCLA_CV_{args.setup}_{args.sampling}/DIR_cls_noCL_{args.mode}/')
    if not os.path.exists(args.saveModel): os.makedirs(args.saveModel)
    '============================================= Main Body of script================================================='

    # Dataset
    assert args.path_list!='', '!!! NO Dataset Sample LIST !!!'
    path_list = args.path_list + f"/data/CV/{args.setup}/"
    # root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'

    testSet = NUCLA_CrossView(root_list=path_list, phase='test',
                              setup=args.setup, dataType=args.dataType,
                              sampling=args.sampling, nClip=args.nClip,
                              T=args.T, maskType=args.maskType)
    testloader = DataLoader(testSet, shuffle=False,
                            batch_size=args.bs, num_workers=args.nw)

    assert args.pretrain!='', '!!! NO Pretrained Dictionary !!!'
    print('pretrain:', args.pretrain)
    stateDict = torch.load(args.pretrain, map_location=args.map_loc)['state_dict']
    weights = torch.load('/home/rsl/CVARDIF/model_weights/dir_cls_noCL_90.pth', map_location=args.map_loc)['state_dict']
    Drr = stateDict['sparseCoding.rr']
    Dtheta = stateDict['sparseCoding.theta']
    


        # Model
    net = Fullclassification(Drr=Drr, Dtheta=Dtheta,
                                fistaLam=args.lam_f, gpu_id=args.gpu_id,
                                num_class=args.num_class, Npole=args.N+1, 
                                dataType=args.dataType, useCL=False,
                                Inference=True,
                                dim=2, group=False, group_reg=0).cuda(args.gpu_id)
        
    net.load_state_dict(weights)
    net.sparseCoding.rr.requires_grad = False
    net.sparseCoding.theta.requires_grad = False

    ACC = []
    print('Experiment config | setup:',args.setup,'sampling:', args.sampling, 'gumbel_thresh:', args.gumbel_thresh,
          '\n\tAlpha(bi):',args.Alpha,'lam1(cls):',args.lam1,'lam2(mse):',args.lam2,
          'lr(mse):',args.lr,'lr_2(cls):',args.lr_2)


    Acc, sparseCode, bcode = testing(testloader, net,
                    args.gpu_id, args.sampling,
                    args.mode, args.withMask,
                    args.gumbel_thresh,
                    None)
    print(f'Acc: {Acc*100:.4f}%')
    ACC.append(Acc)
        
    torch.cuda.empty_cache()
    print('done')
    sparsearray = np.stack([tensor.numpy() for tensor in sparseCode])
    bcodearray = np.stack([code.numpy() for code in bcode])
    np.save('testing_sparse_vectors.npy',sparsearray)
    np.save('testing_binary_code.npy',bcodearray)

if __name__ == "__main__":
    parser = get_parser()
    args=parser.parse_args()
    args.map_loc = "cuda:"+str(args.gpu_id)
    
    main(args)