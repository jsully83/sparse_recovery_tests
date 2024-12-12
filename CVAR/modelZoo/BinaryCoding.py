from scipy.spatial import distance

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torchvision.transforms as transforms

from utils import gridRing
from modelZoo.actRGB import getAttentionRGBFeature, RGBAction
from modelZoo.sparseCoding import DyanEncoder,np
from modelZoo.gumbel_module import GumbelSigmoid

class GroupNorm(nn.Module):
    r"""Applies Group Normalization over a mini-batch of inputs as described in
    the paper `Group Normalization`_ .
    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. The mean and standard-deviation are calculated
    separately over the each group. :math:`\gamma` and :math:`\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.
    This layer uses statistics computed from input data in both training and
    evaluation modes.
    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.
    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)
    Examples::
        # >>> input = torch.randn(20, 6, 10, 10)
        # >>> # Separate 6 channels into 3 groups
        # >>> m = nn.GroupNorm(3, 6)
        # >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        # >>> m = nn.GroupNorm(6, 6)
        # >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        # >>> m = nn.GroupNorm(1, 6)
        # >>> # Activating the module
        # >>> output = m(input)
    .. _`Group Normalization`: https://arxiv.org/abs/1803.08494
    """
    # __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(GroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        return F.group_norm(
            input, self.num_groups, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)


class binaryCoding(nn.Module):
    def __init__(self, num_binary):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(161, 64, kernel_size=(3,3), padding=2),  # same padding
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,1), stride=1),
            # nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(64, 32, kernel_size=(3,3), padding=2),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(1,1), stride=1),
            # nn.BatchNorm2d(num_features=32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

            nn.Conv2d(32, 64, kernel_size=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=(1,1), stride=1),
            # nn.BatchNorm2d(num_features=64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),

        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 500),
            # nn.Linear(64*26*8, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, num_binary)
        )

        for m in self.modules():
            # if m.__class__ == nn.Conv2d or m.__class__ == nn.Linear:
            #     init.xavier_normal(m.weight.data)
            #     m.bias.data.fill_(0)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class binarizeSparseCode(nn.Module):
    def __init__(self, Drr, Dtheta, T, wiRH,
                 gpu_id, Inference=True, fistaLam=0.1):
        super(binarizeSparseCode, self).__init__()
        self.Drr = Drr
        self.Dtheta = Dtheta
        self.Inference = Inference
        self.gpu_id = gpu_id
        self.fistaLam = fistaLam
        self.sparseCoding = DyanEncoder(self.Drr, self.Dtheta, T, wiRH,
                                        lam=fistaLam,
                                        gpu_id=self.gpu_id)
        self.BinaryCoding = GumbelSigmoid()


    def forward(self, x, bi_thresh, inference=True):
        # DYAN encoding
        sparseCode, Dict, _ = self.sparseCoding(x)
        R = torch.matmul(Dict, sparseCode)
        # Gumbel
        binaryCode = self.BinaryCoding(sparseCode**2, bi_thresh, temperature=0.1,
                                       force_hard=True, inference=inference)
        temp = sparseCode*binaryCode
        R_B = torch.matmul(Dict, temp)
        
        return binaryCode, R, sparseCode, R_B


class classificationGlobal(nn.Module):
    def __init__(self, num_class, Npole, dataType, useCL):
        super(classificationGlobal, self).__init__()
        self.num_class = num_class
        self.Npole = Npole
    
        self.useCL = useCL
        self.dataType = dataType
        self.conv1 = nn.Conv1d(self.Npole, 256, 1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm1d(num_features=256, eps=1e-05, affine=True)

        self.conv2 = nn.Conv1d(256, 512, 1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm1d(num_features=512, eps=1e-5, affine=True)

        self.conv3 = nn.Conv1d(512, 1024, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(num_features=1024, eps=1e-5, affine=True)
       
        self.conv4 = nn.Conv2d(self.Npole + 1024, 1024, (3, 1), stride=1, padding=(0, 1))
        self.bn4 = nn.BatchNorm2d(num_features=1024, eps=1e-5, affine=True)

        self.conv5 = nn.Conv2d(1024, 512, (3, 1), stride=1, padding=(0, 1))
        self.bn5 = nn.BatchNorm2d(num_features=512, eps=1e-5, affine=True)

        self.conv6 = nn.Conv2d(512, 256, (3, 3), stride=2, padding=0)
        self.bn6 = nn.BatchNorm2d(num_features=256, eps=1e-5, affine=True)
        if self.dataType == '2D' :
            self.njts = 25
            self.fc = nn.Linear(256*10*2, 1024) #njts = 25
        elif self.dataType == 'rgb':
            self.njts = 512 # for rgb
            # self.fc = nn.Linear(256*61*2, 1024) #for rgb
            self.fc = nn.Linear(256*253*2, 1024) # for att rgb
        elif self.dataType == '2D+rgb':
            self.njts = 512+25
            self.fc = nn.Linear(256*266*2,1024)
            
        self.pool = nn.AvgPool1d(kernel_size=(self.njts))
        # self.fc = nn.Linear(7168,1024) #njts = 34
        
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 128)

        # self.linear = nn.Sequential(nn.Linear(256*10*2,1024),
        #                             nn.LeakyReLU(),
        #                             nn.Linear(1024,512),
        #                             nn.LeakyReLU(),
        #                             nn.Linear(512, 128),
        #                             nn.LeakyReLU())
        if self.useCL == False:
            # self.cls = nn.Linear(128, self.num_class)
            self.cls = nn.Sequential(nn.Linear(128,128),
                                     nn.LeakyReLU(),
                                     nn.Linear(128, self.num_class))
        else:
            self.cls = nn.Sequential(nn.Linear(128, self.num_class))
        self.relu = nn.LeakyReLU()

        'initialize model weights'
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out', nonlinearity='relu' )
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=1)


    def forward(self,x):
        inp = x
        if self.dataType == '2D' or 'rgb':
            dim = 2
        else:
            dim = 3

        bz = inp.shape[0]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        # ipdb.set_trace()
        x_gl = self.pool(self.relu(self.bn3(self.conv3(x))))
        # ipdb.set_trace()
        x_new = torch.cat((x_gl.repeat(1,1,inp.shape[-1]),inp),1).reshape(bz,1024+self.Npole, self.njts,dim)

        x_out = self.relu(self.bn4(self.conv4(x_new)))
        x_out = self.relu(self.bn5(self.conv5(x_out)))
        x_out = self.relu(self.bn6(self.conv6(x_out)))

        'MLP'
        # ipdb.set_trace()
        x_out = x_out.view(bz,-1)  #flatten
        
        x_out = self.relu(self.fc(x_out))
        x_out = self.relu(self.fc2(x_out))
        x_out = self.relu(self.fc3(x_out)) #last feature before cls

        out = self.cls(x_out)

        return out, x_out

class classificationWBinarization(nn.Module):
    def __init__(self, num_class, Npole, num_binary, dataType, useCL):
        super(classificationWBinarization, self).__init__()
        self.num_class = num_class
        self.Npole = Npole
        self.num_binary = num_binary
        self.dataType = dataType
        self.useCL = useCL
        self.BinaryCoding = binaryCoding(num_binary=self.num_binary)
        self.Classifier = classificationGlobal(num_class=self.num_class,
                                               Npole=Npole,
                                               dataType=self.dataType,
                                               useCL=self.useCL)

    def forward(self, x):
        'x is coefficients'
        inp = x.reshape(x.shape[0], x.shape[1], -1).permute(2,1,0).unsqueeze(-1)
        binaryCode = self.BinaryCoding(inp)
        binaryCode = binaryCode.t().reshape(self.num_binary, x.shape[-2], x.shape[-1]).unsqueeze(0)
        label, _ = self.Classifier(binaryCode)

        return label,binaryCode
    
class classificationWBinarizationRGB(nn.Module):
    def __init__(self, num_class, T,  Npole, Drr, Dtheta, dataType, gpu_id):
        super(classificationWBinarizationRGB, self).__init__()
        self.num_class = num_class
        self.Npole = Npole
        self.d_model = 512
        self.lam = 0.1
        self.dataType = dataType
        self.num_binary = 128
        self.seqLen = T
        self.cat = nn.Linear(self.d_model+self.d_model, self.d_model)
        # self.getRGB = RGBAction(num_class=num_class, kinetics_pretrain=kinetics_pretrain)
        self.getRGBFeature = getAttentionRGBFeature(self.d_model, n_head=4,T=self.seqLen)
        self.binaryCoding = binarizeSparseCode(self.num_binary, Drr, Dtheta, gpu_id, Inference=True, fistaLam=self.lam)
        self.Classifier = classificationGlobal(num_class=self.num_class, Npole=Npole,dataType=self.dataType, useCL=False)

    def forward(self, img, roi):
        img_feat = self.getRGBFeature(img) # B x seqLen x d_model
        roi_feat = self.getRGBFeature(roi)
        # ipdb.set_trace()
        feat_cat = torch.cat((img_feat, roi_feat),dim=2).reshape(img_feat.shape[0]*img_feat.shape[1], self.d_model*2)
        # feat_input = self.cat(feat_cat).reshape(img_feat.shape[0], img_feat.shape[1], self.d_model)
        feat_input = feat_cat.reshape(img_feat.shape[0], img_feat.shape[1], self.d_model*2)
        binaryCode, feat_out, sparseCode = self.binaryCoding.forward(feat_input,self.seqLen,bi_thresh=0.500 )
        # ipdb.set_trace()
        label,_ = self.Classifier(sparseCode)

        return label, sparseCode, feat_out, feat_input 


        # label, _, rgbFeat = self.getRGB(img, roi)
        # N, t, dim = rgbFeat.shape
        # binaryCode, reconstruction, sparseCode = self.binaryCoding.forward(rgbFeat,t,bi_thresh=0.5 )
        # ipdb.set_trace()
        # label, _ = self.Classifier(binaryCode)
        # label, _ = self.Classifier(sparseCode)
        # return binaryCode, reconstruction,
        # return label,rgbFeat

class classificationWSparseCode(nn.Module):
    def __init__(self, num_class, Npole, Drr, Dtheta, dataType,dim,fistaLam, gpu_id, useCL):
        super(classificationWSparseCode, self).__init__()
        self.num_class = num_class
        self.Npole = Npole
        # self.Npole = 50
        self.Drr = Drr
        self.Dtheta = Dtheta
        # self.T = T
        self.useCL = useCL
        self.gpu_id = gpu_id
        self.dim = dim
        self.dataType = dataType
        self.Classifier = classificationGlobal(num_class=self.num_class, Npole=self.Npole, dataType=self.dataType, useCL=self.useCL)
        self.fistaLam = fistaLam
        self.sparseCoding = DyanEncoder(self.Drr, self.Dtheta,lam=self.fistaLam, gpu_id=self.gpu_id)
        # self.MasksparseCoding = MaskDyanEncoder(self.Drr, self.Dtheta, lam=fistaLam, gpu_id=self.gpu_id)
        self.groups = np.linspace(0, 160, 161, dtype=np.int)
        group_reg = 0.01
        self.group_regs = torch.ones(len(self.groups)) * group_reg
        # self.sparseCodingGroup = GroupLassoEncoder(self.Drr, self.Dtheta, self.fistaLam, self.groups, self.group_regs, gpu_id)
    def forward(self, x, T):
        # sparseCode, dict, Reconstruction  = self.sparseCoding.forward2(x, T) # w.o. RH
        sparseCode, Dict, Reconstruction = self.sparseCoding(x, T) #RH
        # sparseCode, Reconstruction = self.sparseCodingGroup(x, T) # group lasso
        # print(sparseCode.shape)
        label, lastFeat = self.Classifier(sparseCode)

        return label, Reconstruction, lastFeat
    def forward2(self, x, T, keep_index):
        sparseCode, dict, Reconstruction = self.sparseCoding.thresholdDict(x, T, keep_index)
        label, _ = self.Classifier(sparseCode)
        
        return label, Reconstruction, dict


class classificationWBinarizationRGBDY(nn.Module):
    def __init__(self, pretrain1, pretrain2, num_class,T, Npole, gpu_id):
        super(classificationWBinarizationRGBDY, self).__init__()
        self.pretrain1 = pretrain1
        state_dict1 = torch.load(pretrain1,map_location= "cuda:"+str(gpu_id))['state_dict']
        # ipdb.set_trace()
        Drr1 = state_dict1['sparseCoding.rr']
        Dtheta1 = state_dict1['sparseCoding.theta']
        self.getSkeletonSparseCode = DyanEncoder(Drr1, Dtheta1, 0.1, gpu_id)

        for param in self.getSkeletonSparseCode.parameters():
            param.requires_grad = False

        self.pretrain2 = pretrain2 
        state_dict2 = torch.load(pretrain2,map_location= "cuda:"+str(gpu_id))['state_dict']
        # ipdb.set_trace()
        Drr2 = state_dict2['binaryCoding.sparseCoding.rr']
        Dtheta2 = state_dict2['binaryCoding.sparseCoding.theta']
        self.getRGBSparseCode = classificationWBinarizationRGB(num_class, T=T, Npole=Npole, Drr=Drr2, Dtheta=Dtheta2, dataType='rgb', gpu_id=gpu_id).cuda(gpu_id)
        self.getRGBSparseCode.load_state_dict(state_dict2)
        
        for param in self.getRGBSparseCode.parameters():
            param.requires_grad = False

        self.Classifier = classificationGlobal(num_class=num_class, Npole=Npole,dataType='rgb+dy', useCL=False)

    def forward(self, skeleton,t, img, roi):
        sparseCode1,_,_ = self.getSkeletonSparseCode(skeleton, t)
        _, sparseCode2, _, _ = self.getRGBSparseCode(img, roi)
        sparseCode = torch.cat((sparseCode1, sparseCode2),dim=-1)
        # ipdb.set_trace()
        label,_ = self.Classifier(sparseCode)
        return label
    

class Fullclassification(nn.Module):
    def __init__(self, Drr, Dtheta, T, wiRH,
                 fistaLam, gpu_id,
                 num_class, Npole,
                 dataType, useCL, 
                 Inference):
        super(Fullclassification, self).__init__()
        # DYAN
        self.Drr = Drr
        self.Dtheta = Dtheta
        self.T = T
        self.fistaLam = fistaLam
        self.gpu_id = gpu_id
        # Gumbel
        # self.bi_thresh = 0.505
        # Classifier
        self.num_class = num_class
        self.Npole = Npole
        self.dataType = dataType
        self.useCL = useCL
        # Networks
        self.sparseCoding = DyanEncoder(self.Drr, self.Dtheta, T, wiRH,
                                        lam=fistaLam, gpu_id=self.gpu_id,
                                        freezeD=True)
        self.Inference = Inference
        self.BinaryCoding = GumbelSigmoid()
        self.Classifier = classificationGlobal(num_class=self.num_class,
                                               Npole=self.Npole,
                                               dataType=self.dataType,
                                               useCL=self.useCL)

    def forward(self, x, bi_thresh):
        # DYAN
        sparseCode, Dict, R_C = self.sparseCoding(x) # w.RH
        # Gumbel
        binaryCode = self.BinaryCoding(sparseCode**2, bi_thresh, temperature=0.1,
                                       force_hard=True, inference=self.Inference)
        # Classifier
        label, lastFeat = self.Classifier(binaryCode)
        temp1 = sparseCode * binaryCode
        R_B = torch.matmul(Dict, temp1)

        return label, lastFeat, binaryCode, R_C, R_B


class fusionLayers(nn.Module):
    def __init__(self, num_class, in_chanel_x, in_chanel_y):
        super(fusionLayers, self).__init__()
        self.num_class = num_class
        self.in_chanel_x = in_chanel_x
        self.in_chanel_y = in_chanel_y
        self.cat = nn.Linear(self.in_chanel_x + self.in_chanel_y, 128)
        self.cls = nn.Linear(128, self.num_class)
        self.relu = nn.LeakyReLU()
    def forward(self, feat_x, feat_y):
        twoStreamFeat = torch.cat((feat_x, feat_y), 1)
        out = self.relu(self.cat(twoStreamFeat))
        label = self.cls(out)
        return label, out


class twoStreamClassification(nn.Module):
    def __init__(self, num_class, Npole, num_binary, Drr, Dtheta, dim, gpu_id, inference, fistaLam, dataType, kinetics_pretrain):
        super(twoStreamClassification, self).__init__()
        self.num_class = num_class
        self.Npole = Npole
        self.num_binary = num_binary
        self.Drr = Drr
        self.Dtheta = Dtheta
        # self.PRE = PRE
        self.gpu_id = gpu_id
        self.dataType = dataType
        self.dim = dim
        self.kinetics_pretrain = kinetics_pretrain
        self.Inference = inference
        self.nClip = 1
        self.fistaLam = fistaLam
        self.withMask = False

        # self.dynamicsClassifier = Fullclassification(self.num_class, self.Npole,
        #                         self.Drr, self.Dtheta, self.dim, self.dataType, self.Inference, self.gpu_id, self.fistaLam,self.withMask)

        # self, dim_embed, Npole, Drr, Dtheta, Inference, gpu_id, dim, mode, fistaLam,fineTune,useCL, nClip
        self.dynamicsClassifier = contrastiveNet(dim_embed=128, Npole=self.Npole, Drr=self.Drr, Dtheta=self.Dtheta, Inference=True, gpu_id=self.gpu_id, dim=2, mode='2D', dataType=self.dataType, fistaLam=fistaLam, fineTune=True, useCL=False, nClip=self.nClip)
        self.RGBClassifier = RGBAction(self.num_class, self.kinetics_pretrain)

        self.lastPred = fusionLayers(self.num_class, in_chanel_x=512, in_chanel_y=128)

    def forward(self,skeleton, image, rois, fusion, bi_thresh):
        # stream = 'fusion'
        bz = skeleton.shape[0]
        if bz == 1:
            skeleton = skeleton.repeat(2,1,1,1)
            image = image.repeat(2,1,1,1,1)
            rois = rois.repeat(2,1,1,1,1)
        label1,lastFeat_DIR, binaryCode, Reconstruction = self.dynamicsClassifier(skeleton, bi_thresh)
        label2, lastFeat_CIR,_ = self.RGBClassifier(image, rois)

        if fusion:
            label = {'RGB':label1, 'Dynamcis':label2}
            feats = lastFeat_DIR
        else:
            # label = 0.5 * label1 + 0.5 * label2
            label, feats= self.lastPred(lastFeat_DIR, lastFeat_CIR)
        if bz == 1 :
            nClip = int(label.shape[0]/2)
            return label[0:nClip], binaryCode[0:nClip], Reconstruction[0:nClip], feats
        else:
            return label, binaryCode, Reconstruction, feats


class MLP(nn.Module):
    def __init__(self,  dim_in):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(dim_in, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.layer2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        # self.gelu = nn.GELU()
        self.relu = nn.LeakyReLU()
        self.sig = nn.Sigmoid()

    def forward(self,x):
        x_out = self.relu(self.bn1(self.layer1(x)))
        x_out = self.relu(self.bn2(self.layer2(x_out)))
        # x_out = self.sig(x_out)

        return x_out

class contrastiveNet(nn.Module):
    def __init__(self, dim_embed, Npole, Drr, Dtheta, fistaLam,
                 mode, Inference, fineTune, useCL, 
                 dim, dataType, nClip, gpu_id):
        super(contrastiveNet, self).__init__()

        # self.dim_in = dim_in
        self.Npole = Npole
        self.nClip = nClip
        self.dim_embed = dim_embed
        self.Drr = Drr
        self.Dtheta = Dtheta
        self.Inference = Inference
        self.gpu_id = gpu_id
        self.dim_data = dim
        self.mode = mode
        self.fistaLam = fistaLam
        # self.withMask = False
        self.useGroup = False
        self.dataType = dataType
        self.group_reg = 0.01
        self.num_class = 10
        self.fineTune = fineTune
        self.useCL = useCL
        if self.mode == 'rgb':
            self.backbone = RGBAction(num_class=self.num_class, kinetics_pretrain='./pretrained/i3d_kinetics.pth')
            dim_mlp = self.backbone.cls.in_features
        else:
            self.backbone = Fullclassification(self.num_class, self.Npole, self.Drr, self.Dtheta,
                                               self.dim_data, self.dataType, self.Inference, 
                                               self.gpu_id, self.fistaLam,
                                               self.useGroup, self.group_reg,
                                               self.useCL)
        # if self.useCL == False:
        #     dim_mlp = self.backbone.Classifier.cls.in_features
        # else:
            dim_mlp = self.backbone.Classifier.cls[0].in_features
        self.proj = nn.Linear(dim_mlp,self.dim_embed)
        self.relu = nn.LeakyReLU()
        

    def forward(self, x, y):
        'if x: affine skeleton, then y:bi_thresh'
        'if x: img, then y: roi'
        bz = x.shape[0]
        # if len(x.shape) == 3:
        #     x = x.unsqueeze(0)
        if self.fineTune == False:
            if self.mode == 'rgb':
                if bz < 2:
                    x = x.repeat(2, 1, 1, 1, 1, 1)
                    bz = x.shape[0]
                x1_img, x2_img = x[:,0], x[:,1]
                x1_roi, x2_roi = x[:,2], x[:,3]
                _, lastFeat1, _ = self.backbone(x1_img, x1_roi)
                _, lastFeat2, _ = self.backbone(x2_img, x2_roi)
            else:
                if_multi = True if x.ndim > 4 else False
                if bz < 2:
                    x = x.repeat(2, 1, 1, 1, 1) if if_multi else x.repeat(2, 1, 1, 1)
                    bz = x.shape[0]
                # x = x.reshape(x.shape[0]* x.shape[1], x.shape[2], x.shape[3])
                x1 = x[:,0]
                x2 = x[:,1]
                _, lastFeat1, _, _ = self.backbone(x1, y)
                _, lastFeat2, _, _ = self.backbone(x2, y)
            embedding1 = self.relu(self.proj(lastFeat1))
            embedding2 = self.relu(self.proj(lastFeat2))
            embed1 = torch.mean(embedding1.reshape(bz, self.nClip, embedding1.shape[-1]),1)
            embed2 = torch.mean(embedding2.reshape(bz, self.nClip, embedding2.shape[-1]),1)
            z1 = F.normalize(embed1, dim=1)
            z2 = F.normalize(embed2, dim=1)

            features = torch.cat([z1,z2], dim=0)
            labels = torch.cat([torch.arange(bz) for i in range(2)], dim=0)
            labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().cuda(self.gpu_id)

            simL_matrix = torch.matmul(features, features.T)
            mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda(self.gpu_id)
            labels = labels[~mask].view(labels.shape[0],-1)
            simL_matrix = simL_matrix[~mask].view(simL_matrix.shape[0], -1)
            positives = simL_matrix[labels.bool()].view(labels.shape[0], -1)
            negatives = simL_matrix[~labels.bool()].view(simL_matrix.shape[0], -1)

            logits = torch.cat([positives, negatives], dim=1)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(self.gpu_id)
            temper = 0.07 #default
            logits = logits/temper

            return logits, labels
        else:
            if self.mode == 'rgb':
                return self.backbone(x, y)
            else:
                return self.backbone(x, y)


if __name__ == '__main__':
    gpu_id = 0

    N = 2*80
    P, Pall = gridRing(N)
    Drr = abs(P)
    Drr = torch.from_numpy(Drr).float()
    Dtheta = np.angle(P)
    Dtheta = torch.from_numpy(Dtheta).float()
    kinetics_pretrain = '../pretrained/i3d_kinetics.pth'
    """"
    net = twoStreamClassification(num_class=10, Npole=161, num_binary=161, Drr=Drr, Dtheta=Dtheta,
                                  dim=2, gpu_id=gpu_id,inference=True, fistaLam=0.1, dataType='2D',
                                  kinetics_pretrain=kinetics_pretrain).cuda(gpu_id)
    x = torch.randn(5, 36, 50).cuda(gpu_id)
    xImg = torch.randn(5, 20, 3, 224, 224).cuda(gpu_id)
    T = x.shape[1]
    xRois = xImg
    label, _, _ = net(x, xImg, xRois, T, False)

    """""
    
    


    print('check')






