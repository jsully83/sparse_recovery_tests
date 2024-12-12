from pathlib import Path
import time
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
import numpy as np
# import matplotlib.pyplot as plt
import random
from einops import rearrange

from crossView_UCLA_ske import NUCLA_CrossView
from Dyan import Dyan




#globals to be added to config later
setup = 'setup1'
dataType = '2D'
sampling = 'Single'
nClip = 1
maskType = 'None'
nw = 8
learning_rate = 1.0e-4
n_epochs = 100
seed = 1337

# Set seed for PyTorch (CPU)
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Set seed for PyTorch (GPU)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you use multiple GPUs

# Make PyTorch operations deterministic (slows down computation but improves reproducibility)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
print(f'Pytorch Version: {torch.__version__} \n {torch.cuda.get_device_name()}')

test_name = 'default'
output_path = Path('/home/rsl/sparse_recovery_tests/tests').joinpath(test_name)
output_path.mkdir(parents=True, exist_ok=True)

data_files_list = Path().home().joinpath('data', 'N-UCLA_MA_3D')
if data_files_list.exists():
        data_files_list = data_files_list.as_posix() + f"/data/CV/{setup}/"
else:
    raise FileExistsError

trainSet = NUCLA_CrossView(root_list=data_files_list, phase='train',
                            setup=setup, dataType=dataType,
                            sampling=sampling, nClip=nClip,
                            T=36, maskType=maskType) 
trainLoader = DataLoader(trainSet, shuffle=True,
                             batch_size=8, num_workers=nw)

testSet = NUCLA_CrossView(root_list=data_files_list, phase='test',
                            setup=setup, dataType=dataType,
                            sampling=sampling, nClip=nClip,
                            T=36, maskType=maskType) 
testLoader = DataLoader(trainSet, shuffle=False,
                             batch_size=8, num_workers=nw)

dyan = Dyan(device, config_filename='dyan_config', train_new_dict=True)
dyan.to(device)

# for name, param in dyan.named_parameters():
#     if param.requires_grad:
#         print(f"Parameter name: {name}, requires_grad={param.requires_grad}, value: {param}")

mseloss = torch.nn.MSELoss()

optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, dyan.parameters()), 
                            lr=learning_rate, weight_decay=0.001, momentum=0.9)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 70], gamma=0.1)

train_loss_list = []
with torch.autograd.set_detect_anomaly(True):
    for epoch in range(n_epochs):
        start = time.perf_counter()
        for i, sample in enumerate(trainLoader):
            print(i)
            # setup batches
            skeletons = sample['input_skeletons']['normSkeleton'].float().to(device)
            t = skeletons.shape[1] # (batch_size x num_clips) x t x num_joint x dim_joint
            gt_skeletons = skeletons.reshape(skeletons.shape[0], t, -1).to(device)
            
            # train
            optimizer.zero_grad()
            dyan.train()
            pred_skeletons, C_pred = dyan(gt_skeletons)
            
            loss = mseloss(pred_skeletons, gt_skeletons)

            loss.backward()
            optimizer.step()

        stop = time.perf_counter()
        elapsed_time = stop - start
        seconds = int(elapsed_time)
        milliseconds = int((elapsed_time - seconds) * 1000)
        train_loss = np.mean(np.asarray(train_loss_list))
        print(f'epoch: {epoch} - train loss: {train_loss} elapsed time: {seconds}.{milliseconds:03d} seconds')

