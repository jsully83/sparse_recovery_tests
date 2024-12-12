# DIR-4: fine-tune CVAR cls with CL feature, N-UCLA dataset
import time
import argparse
from matplotlib import pyplot as plt

import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dataset.crossView_UCLA_ske import os, np, random, NUCLA_CrossView
from modelZoo.BinaryCoding import nn, gridRing,contrastiveNet
from utils import load_fineTune_model
from test_cls_CV_DIR import testing, getPlots

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

def get_parser():
    def str2bool(v):
        if    v.lower() in ('yes', 'true', 't', 'y', '1'):  return True
        elif  v.lower() in ('no', 'false', 'f', 'n', '0'):  return False
        else:  raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
    parser = argparse.ArgumentParser(description='CVARDIF')
    parser.add_argument('--modelRoot', default='/data/Dan/202111_CVAR/NUCLA/CV_DIR_cls_ft',
                        help='the work folder for storing experiment results')
    parser.add_argument('--path_list', default='', help='')
    parser.add_argument('--pretrain',  default='', help='')
    parser.add_argument('--cus_n', default='', help='customized name')
    
    parser.add_argument('--dataset', default='NUCLA', help='') # dataset = 'NUCLA'
    parser.add_argument('--setup', default='setup1', help='') # setup = 'setup1' # v1,v2 train, v3 test;
    parser.add_argument('--num_class', default=10, type=int, help='') # num_class = 10
    parser.add_argument('--dataType', default='2D', help='') # dataType = '2D'
    parser.add_argument('--sampling', default='Single', help='') # sampling = 'Multi' #sampling strategy
    parser.add_argument('--nClip', default=1, type=int, help='') # sampling=='multi' or sampling!='Single'
    parser.add_argument('--bs', default=8, type=int, help='')
    parser.add_argument('--nw', default=8, type=int, help='')
    
    parser.add_argument('--mode', default='dy+bi+cl', help='dy+bi+cl | dy+cl | rgb+dy') # mode = 'dy+bi+cl'
    parser.add_argument('--RHdyan', default='1', type=str2bool, help='') # RHdyan = True
    parser.add_argument('--withMask', default='0', type=str2bool, help='') # withMask = False
    parser.add_argument('--maskType', default='None', help='') # maskType = 'score'
    parser.add_argument('--contrastive', default='1', type=str2bool, help='') # constrastive = True
    parser.add_argument('--finetune', default='1', type=str2bool, help='') 
    parser.add_argument('--fusion', default='0', type=str2bool, help='') # fusion = False
    parser.add_argument('--groupLasso', default='0', type=str2bool, help='')

    parser.add_argument('--T', default=36, type=int, help='') # T = 36 # input clip length
    parser.add_argument('--N', default=80*2, type=int, help='') # N = 80*2
    parser.add_argument('--lam_f', default=0.1, type=float) # fistaLam = 0.1
    parser.add_argument('--gumbel_thresh', default=0.505, type=float) # gumbel_thresh = 0.505

    parser.add_argument('--gpu_id', default=1, type=int, help='') # gpu_id = 5
    parser.add_argument('--Epoch', default=50, type=int, help='') # Epoch = 100
    parser.add_argument('--lr', default=1e-4, type=float, help='sparse coding') # lr = 1e-3 # classifier
    parser.add_argument('--lr_2', default=5e-4, type=float, help='classifier') # lr_2 = 1e-4  # sparse codeing
    parser.add_argument('--Alpha', default=0.1, type=float, help='bi loss')  # Alpha = 0.1
    parser.add_argument('--lam1', default=1, type=float, help='cls loss')     # lam1  = 2
    parser.add_argument('--lam2', default=0.5, type=float, help='mse loss')   # lam2  = 1

    return parser

def main(args):
    args.saveModel = os.path.join(args.modelRoot,
                                  f'NUCLA_CV_{args.setup}_{args.sampling}/DIR_cls_ft_{args.mode}/')
    if not os.path.exists(args.saveModel): os.makedirs(args.saveModel)
    print('mode:',args.mode, 'model path:', args.saveModel, 'gpu:', args.gpu_id)
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
    testloader = DataLoader(testSet, shuffle=True,
                            batch_size=args.bs, num_workers=args.nw)

    net = contrastiveNet(dim_embed=128, Npole=args.N+1, 
                         Drr=Drr, Dtheta=Dtheta, fistaLam=args.lam_f,
                         mode=args.mode, Inference=True,
                         dataType=args.dataType, dim=2, nClip=args.nClip, 
                         fineTune=args.finetune, useCL=args.contrastive,
                         gpu_id=args.gpu_id).cuda(args.gpu_id)

    # 'load pre-trained contrastive model'
    #pre_train = modelRoot + sampling + '/' + mode + '/T36_contrastive_fineTune_all/' + '40.pth'        
    assert args.pretrain!='', '!!! NO Pretrained Dictionary !!!'
    print('pretrain:', args.pretrain)
    state_dict = torch.load(args.pretrain, map_location=args.map_loc)
    net = load_fineTune_model(state_dict, net)
    net.train()
    # frozen all layers except last classification layer
    for p in net.backbone.sparseCoding.parameters():       p.requires_grad = False
    for p in net.backbone.Classifier.parameters():         p.requires_grad = True
    # for p in net.backbone.Classifier.parameters():         p.requires_grad = False
    # for p in net.backbone.Classifier.cls[-1].parameters(): p.requires_grad = True

    optimizer = torch.optim.SGD(
            [{'params': filter(lambda x: x.requires_grad,net.backbone.sparseCoding.parameters()),
              'lr':args.lr},
            {'params': filter(lambda x: x.requires_grad, net.backbone.Classifier.parameters()),
             'lr': args.lr_2}], weight_decay=1e-3, momentum=0.9)

    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)
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
    print('RHdyan:',args.RHdyan, 'useCL:', args.contrastive, 'fineTune:', args.finetune )
    for epoch in range(1, args.Epoch+1):
        print('start training epoch:', epoch)
        lossVal = []
        lossCls = []
        lossBi = []
        lossMSE = []
        start_time = time.time()
        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()

            skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(args.gpu_id)
            # skeletons = sample['input_skeletons']['unNormSkeleton'].float().cuda(gpu_id)
            # skeletons = sample['input_skeletons']['affineSkeletons'].float().cuda(gpu_id)
            visibility = sample['input_skeletons']['visibility'].float().cuda(args.gpu_id)
            gt_label = sample['action'].cuda(args.gpu_id)

            if args.sampling == 'Single':
                t = skeletons.shape[1]
                input_skeletons = skeletons.reshape(skeletons.shape[0], t, -1)  #bz, T, 25, 2
                input_mask = visibility.reshape(visibility.shape[0], t, -1)
                nClip = 1
            else:
                t = skeletons.shape[2]
                input_skeletons = skeletons.reshape(skeletons.shape[0],skeletons.shape[1], t, -1) #bz,clip, T, 25, 2 --> bz*clip, T, 50
                # input_mask = visibility.reshape(visibility.shape[0]*visibility.shape[1], t, -1)
                nClip = skeletons.shape[1]

            actPred, lastFeat, binaryCode, output_skeletons = net(input_skeletons, args.gumbel_thresh)
            bi_gt = torch.zeros_like(binaryCode).cuda(args.gpu_id)
            actPred = actPred.reshape(skeletons.shape[0], nClip, args.num_class)
            actPred = torch.mean(actPred, 1)
            # target_skeletons = skeletons.reshape(skeletons.shape[0]*skeletons.shape[1],t,-1)
            # loss = args.lam1 * Criterion(actPred, gt_label) \
            #        + args.lam2 * mseLoss(output_skeletons, target_skeletons.squeeze(-1)) \
            #        + args.Alpha * L1loss(binaryCode, bi_gt)
            # lossMSE.append(mseLoss(output_skeletons, target_skeletons.squeeze(-1)).data.item())
            loss = args.lam1 * Criterion(actPred, gt_label) \
                   + args.lam2 * mseLoss(output_skeletons, input_skeletons) \
                   + args.Alpha * L1loss(binaryCode, bi_gt)

            
            lossBi.append(L1loss(binaryCode, bi_gt).data.item())
            lossMSE.append(mseLoss(output_skeletons, input_skeletons.squeeze(-1)).data.item())
            loss.backward()
            # print('rr.grad:', net.sparseCoding.rr.grad, 'mse:', lossMSE[-1])
            optimizer.step()
            lossVal.append(loss.data.item())
            lossCls.append(Criterion(actPred, gt_label).data.item())

        loss_val = np.mean(np.array(lossVal))
        LOSS.append(loss_val)
        LOSS_CLS.append(np.mean(np.array((lossCls))))
        LOSS_MSE.append(np.mean(np.array(lossMSE)))
        LOSS_BI.append(np.mean(np.array(lossBi)))
        print('epoch:', epoch, '|loss:', loss_val, '|cls:', np.mean(np.array(lossCls)),
              '|mse:', np.mean(np.array(lossMSE)), '|bi:', np.mean(np.array(lossBi)))
        end_time = time.time()
        print('training time(h):', (end_time - start_time) / 3600)

        scheduler.step()
        if epoch % 5 == 0:
            torch.save({'state_dict': net.state_dict(),
                       'optimizer': optimizer.state_dict()},
                       args.saveModel +'dir_cls_ft'+ str(epoch) + '.pth')

            Acc = testing(testloader, net, args.gpu_id, 
                          args.sampling, args.mode, 
                          args.withMask, args.gumbel_thresh, None)
            print('testing epoch:', epoch, f'Acc: {Acc*100:.4f}%')
            ACC.append(Acc)


if __name__ == "__main__":
    parser = get_parser()
    args=parser.parse_args()
    args.map_loc = "cuda:"+str(args.gpu_id) # map_loc = "cuda:"+str(gpu_id)
    if args.sampling == 'Single': args.nClip = 1
    else:                         args.nClip = 6
    main(args)
    # 'plotting results:'
    # getPlots(LOSS,LOSS_CLS, LOSS_MSE, LOSS_BI, ACC,fig_name='DY_CL.pdf')
    torch.cuda.empty_cache()
    print('done')