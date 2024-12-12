from einops import rearrange
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from dataset.crossView_UCLA import np, torch, NUCLA_CrossView
from modelZoo.BinaryCoding import gridRing, twoStreamClassification, contrastiveNet
from utils import gridRing

def testing(dataloader, net,
            gpu_id, sampling, mode, 
            withMask, gumbel_thresh,
            keep_index):
    count = 0
    pred_cnt = 0

    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            # print('sample:', i)
            skeletons = sample['input_skeletons']['normSkeleton'].float().cuda(gpu_id)
            gt = sample['action'].cuda(gpu_id)

            Nsample, nClip, t = skeletons.shape[0], skeletons.shape[1], skeletons.shape[2]
            input_skeletons =rearrange(skeletons, 'b c t n d -> (b c) t (n d)')
            if mode == 'dy+bi+cl':
                # input_skeletons = skeletons.reshape(skeletons.shape[0]*skeletons.shape[1], t, -1)
                actPred, _,biCode, _,_ = net(input_skeletons, gumbel_thresh)
            else:
                # actPred, _ , _= net(input_skeletons, t)
                actPred, _,_ = net.forward2(input_skeletons, t, keep_index)

            actPred = actPred.reshape(Nsample, nClip, actPred.shape[-1])
            actPred = torch.mean(actPred, 1)
            pred = torch.argmax(actPred, 1)

            correct = torch.eq(gt, pred).int()
            count += gt.shape[0]
            pred_cnt += torch.sum(correct).data.item()

        Acc = pred_cnt/count

    return Acc

def getPlots(LOSS,LOSS_CLS, LOSS_MSE, LOSS_BI, ACC, fig_name):
    'x-axis: number of epochs'
    colors = ['#1f77b4',
              '#ff7f0e',
              '#2ca02c',
              '#d62728',
              '#9467bd',
              '#8c564b',
              '#e377c2',
              '#7f7f7f',
              '#bcbd22',
              '#17becf',
              '#1a55FF']

    fig, axs = plt.subplots(2,1)
    N = len(LOSS)
    axs[0].plot(N, LOSS, 'r-*', label='total loss')
    axs[0].plot(N, LOSS_CLS, 'b-*', label='cls loss')
    axs[0].plot(N, LOSS_MSE, 'g-*', label='mse loss')

    axs[0].set_title('loss v.s epoch')
    axs[0].set_xlabel('number of epochs')
    axs[0].set_ylabel('loss val')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(N, ACC, 'r-+', label='accuracy')
    axs[1].lagend()
    axs[1].set_xlabel('number of epochs')
    axs[1].set_ylabel('accuracy')
    axs[1].set_title('classification accuracy v.s epoch')
    axs[1].grid(True)

    fig.tight_layout()
    fname = './figs/' + fig_name
    plt.savefig(fname)

if __name__ == "__main__":
    gpu_id = 5
    bz = 8
    num_workers = 2
    'initialized params'
    N = 80 * 2
    P, Pall = gridRing(N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()

    mode = 'dy+bi+cl'
    T = 36
    dataset = 'NUCLA'
    sampling = 'Multi'
    withMask = False
    gumbel_thresh = 0.505
    setup = 'setup1'
    path_list = './data/CV/' + setup + '/'
    testSet = NUCLA_CrossView(root_list=path_list, dataType='2D', sampling=sampling, phase='test', cam='2,1', T=T,
                              maskType='score', setup=setup)
    testloader = DataLoader(testSet, batch_size=bz, shuffle=True, num_workers=num_workers)

    if mode == 'dy+bi+cl':

        # net = Fullclassification(num_class=10, Npole=(N + 1), Drr=Drr, Dtheta=Dtheta, dim=2, dataType='2D',
        #                          Inference=True,
        #                          gpu_id=gpu_id, fistaLam=0.1, group=False, group_reg=0.01).cuda(gpu_id)
        net = contrastiveNet(dim_embed=128, Npole=N + 1, Drr=Drr, Dtheta=Dtheta, Inference=True, gpu_id=gpu_id, dim=2,
                               dataType='2D', fistaLam=0.1, fineTune=True).cuda(gpu_id)
    else:
        kinetics_pretrain = './pretrained/i3d_kinetics.pth'
        net = twoStreamClassification(num_class=10, Npole=(N + 1), num_binary=(N + 1), Drr=Drr, Dtheta=Dtheta, dim=2,
                                  gpu_id=gpu_id, inference=True, fistaLam=0.1, dataType='2D',
                                  kinetics_pretrain=kinetics_pretrain).cuda(gpu_id)

    # dy_pretrain = modelRoot + sampling + '/dy+bi+cl/T36_contrastive_fineTune_fixed/40.pth'
    # stateDict = torch.load(dy_pretrain, map_location=map_loc)
    # net.dynamicsClassifier = load_pretrainedModel(stateDict, net.dynamicsClassifier)
    ckpt = '/home/yuexi/Documents/ModelFile/crossView_NUCLA/' + sampling + '/' + mode + '/T36_contrastive_fineTune_fixed/' + '40.pth'
    stateDict = torch.load(ckpt, map_location="cuda:" + str(gpu_id))['state_dict']
    net.load_state_dict(stateDict)

    Acc = testing(testloader,net, gpu_id, sampling, mode, withMask,gumbel_thresh)

    print('Acc:%.4f' % Acc)
    print('done')
