# DIR-2: train CVAR, N-UCLA dataset
import time
import ipdb
from einops import rearrange
import argparse
from matplotlib import pyplot as plt
# from ptflops import get_model_complexity_info

import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dataset.crossView_UCLA_ske import os, np, random, NUCLA_CrossView
from modelZoo.BinaryCoding import Fullclassification, nn, gridRing, classificationWSparseCode
from test_cls_CV_DIR import testing

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
    parser.add_argument('--bs', default=32, type=int, help='')
    parser.add_argument('--nw', default=8, type=int, help='')
    
    parser.add_argument('--mode', default='dy+bi+cl', help='dy+bi+cl | dy+cl | rgb+dy')
    parser.add_argument('--wiRH', default='1', type=str2bool, help='')
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
    parser.add_argument('--lr_2', default=1e-4, type=float, help='classifier')
    parser.add_argument('--Alpha', default=0.1, type=float, help='bi loss')
    parser.add_argument('--lam1', default=1, type=float, help='cls loss')
    parser.add_argument('--lam2', default=0.5, type=float, help='mse loss')

    return parser

def main(args):
    str_conf = f"{'wiRH' if args.wiRH else 'woRH'} "
    print(f" {args.mode} | {str_conf} | Batch Size: Train {args.bs} | Test {args.bs} ")
    print(f"\tlam_f: {args.lam_f} | Alpha: {args.Alpha} | lam2: {args.lam2} | lr_2: {args.lr_2} | g_t: {args.gumbel_thresh}")
    args.saveModel = os.path.join(args.modelRoot,
                                  f'NUCLA_CV_{args.setup}_{args.sampling}/DIR_cls_noCL_{str_conf}/')
    if not os.path.exists(args.saveModel): os.makedirs(args.saveModel)
    '============================================= Main Body of script================================================='
    P,Pall = gridRing(args.N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()
    # Dataset
    assert args.path_list!='', '!!! NO Dataset Sample LIST !!!'
    path_list = args.path_list + f"/data/CV/{args.setup}/"
    # root_skeleton = '/data/Dan/N-UCLA_MA_3D/openpose_est'
    trainSet = NUCLA_CrossView(root_list=path_list, phase='train',
                               setup=args.setup, dataType=args.dataType,
                               sampling=args.sampling, nClip=args.nClip,
                               T=args.T, maskType=args.maskType) 
    trainloader = DataLoader(trainSet, shuffle=True,
                             batch_size=args.bs, num_workers=args.nw)
    testSet = NUCLA_CrossView(root_list=path_list, phase='test',
                              setup=args.setup, dataType=args.dataType,
                              sampling=args.sampling, nClip=args.nClip,
                              T=args.T, maskType=args.maskType)
    testloader = DataLoader(testSet, shuffle=False,
                            batch_size=args.bs, num_workers=args.nw)

    if args.mode == 'dy+bi+cl':
        # rhDYAN+bi+cl
        assert args.pretrain!='', '!!! NO Pretrained Dictionary !!!'
        print('pretrain:', args.pretrain)
        stateDict = torch.load(args.pretrain, map_location=args.map_loc)['state_dict']
        Drr = stateDict['sparseCoding.rr']
        Dtheta = stateDict['sparseCoding.theta']
        # Model
        net = Fullclassification(Drr, Dtheta, args.T, args.wiRH,
                                 args.lam_f, args.gpu_id,
                                 args.num_class, args.N+1, 
                                 args.dataType, useCL=False,
                                 Inference=True).cuda(args.gpu_id)
        total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")
        total_params = sum(p.numel() for p in net.parameters())
        print(f"Total parameters (including non-trainable): {total_params}")
        # Freeze the Dictionary part
        net.train()
        net.sparseCoding.rr.requires_grad = False
        net.sparseCoding.theta.requires_grad = False
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()),
                                    lr=args.lr_2, weight_decay=0.001, momentum=0.9)
    elif args.mode == 'dy+cl':
        # NOTE: this mode is only for regular DYAN so far
        # pretrain = './pretrained/NUCLA/' + setup + '/' + sampling + '/pretrainedDyan.pth'
        # pretrain = '/home/yuexi/Documents/ModelFile/crossView_NUCLA/Single/regularDYAN_seed123/60.pth'
        pretrain = '/home/yuexi/Documents/ModelFile/crossView_NUCLA/Single/rhDYAN_bi/100.pth'
        print('pretrain:', pretrain)
        stateDict = torch.load(pretrain, map_location=args.map_loc)['state_dict']
        Drr = stateDict['sparseCoding.rr'] # for RH
        Dtheta = stateDict['sparseCoding.theta']
        net = classificationWSparseCode(num_class=args.num_class,
                                        Npole=args.N+1,
                                        Drr=Drr, Dtheta=Dtheta,
                                        dataType=args.dataType, dim=2,
                                        fistaLam=args.lam_f,
                                        gpu_id=args.gpu_id,
                                        useCL=False).cuda(args.gpu_id)
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, net.parameters()),
                                    lr=args.lr, weight_decay=0.001, momentum=0.9)
        
        net.sparseCoding.rr.requires_grad = False
        net.sparseCoding.theta.requires_grad = False
    
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[25, 40], gamma=0.1)
    Criterion = torch.nn.CrossEntropyLoss()
    mseLoss = torch.nn.MSELoss()
    L1loss = torch.nn.SmoothL1Loss()

    LOSS = []
    ACC = []
    LOSS_CLS = []
    LOSS_MSE = []
    LOSS_BI = []
    print('Experiment config | setup:',args.setup,'sampling:', args.sampling, 'gumbel_thresh:', args.gumbel_thresh,
          '\n\tAlpha(bi):',args.Alpha,'lam1(cls):',args.lam1,'lam2(mse):',args.lam2,
          'lr(mse):',args.lr,'lr_2(cls):',args.lr_2)
    for epoch in range(1, args.Epoch+1):
        print('start training epoch:', epoch)
        # net.train()

        lossVal = []
        lossCls = []
        lossBi = []
        lossMSE = []
        start_time = time.time()
        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            # batch_size, num_clips, t, num_joint, dim_joint
            skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(args.gpu_id)
            gt_label = sample['action'].cuda(args.gpu_id)

            Nsample, nClip, t = skeletons.shape[0], skeletons.shape[1], skeletons.shape[2]
            input_skeletons =rearrange(skeletons, 'b c t n d -> (b c) t (n d)')
            if args.mode == 'dy+bi+cl':
                keep_index = None
                actPred, _, binaryCode, output_skeletons, _ = net(input_skeletons, args.gumbel_thresh) 
                actPred = actPred.reshape(Nsample, nClip, args.num_class)
                actPred = torch.mean(actPred, 1)
                
                loss = args.lam1*Criterion(actPred, gt_label)
                bi_gt = torch.zeros_like(binaryCode).cuda(args.gpu_id)
                lossMSE.append(mseLoss(output_skeletons, input_skeletons).data.item())
                lossBi.append(L1loss(binaryCode, bi_gt).data.item())
            elif args.mode == 'dy+cl':
                # NOTE: dy+cl
                keep_index = None
                actPred, output_skeletons, _ = net(input_skeletons, t) #bi_thresh=gumbel threshold
                # actPred, output_skeletons,_ = net.forward2(input_skeletons, t, keep_index)
                actPred = actPred.reshape(skeletons.shape[0], nClip, args.num_class)
                actPred = torch.mean(actPred, 1)

                # loss = lam1 * Criterion(actPred, gt_label) + lam2 * mseLoss(output_skeletons, input_skeletons)
                loss = Criterion(actPred, gt_label) #'with fixed dyan'
                lossMSE.append(mseLoss(output_skeletons, input_skeletons).data.item())
                lossBi.append(0)

            loss.backward()
            # ipdb.set_trace()
            optimizer.step()
            lossVal.append(loss.data.item())
            lossCls.append(Criterion(actPred, gt_label).data.item())
        loss_val = np.mean(np.array(lossVal))
        # LOSS.append(loss_val)
        # LOSS_CLS.append(np.mean(np.array((lossCls))))
        # LOSS_MSE.append(np.mean(np.array(lossMSE)))
        # LOSS_BI.append(np.mean(np.array(lossBi)))
        # print('epoch:', epoch, '|loss:', loss_val, '|cls:', np.mean(np.array(lossCls)), '|mse:', np.mean(np.array(lossMSE)), '|bi:', np.mean(np.array(lossBi)))
        print('epoch:', epoch, 'loss:', loss_val)
        end_time = time.time()
        print('training time(h):', (end_time - start_time)/3600)
        scheduler.step()
        
        if epoch % 1 == 0:
            if epoch % 5 ==0 :
                torch.save({'state_dict': net.state_dict(),
                   'optimizer': optimizer.state_dict()}, 
                   args.saveModel + f"dir_cls_noCL_{epoch}.pth")
            net.eval()
            Acc = testing(testloader, net,
                          args.gpu_id, args.sampling,
                          args.mode, args.withMask,
                          args.gumbel_thresh,
                          keep_index)
            print('testing epoch:',epoch, f'Acc: {Acc*100:.4f}%')
            ACC.append(Acc)
        
    torch.cuda.empty_cache()
    print('done')

if __name__ == "__main__":
    parser = get_parser()
    args=parser.parse_args()
    args.map_loc = "cuda:"+str(args.gpu_id)
    
    main(args)