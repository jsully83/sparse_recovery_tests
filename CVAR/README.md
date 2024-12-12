# Official Implementation for CVAR

## Environment
- python=3.7
- PyTorch=1.6.0

## Original [Github Repo](https://northeastern-my.sharepoint.com/:u:/g/personal/luo_dan1_northeastern_edu/EdzEbgrHE-1DocUc7IdQ6-EBekyUWyDZt-wXyw5fRCQPLg?email=camps%40coe.neu.edu&e=M2cv2I) from Yuexi Zhang

## Dataset:
[Northwestern-UCLA Multiview Action 3D Dataset(NUCLA)](https://wangjiangb.github.io/my_data.html) \
[List]() for Cross View setup.

## Training:
Almost all hyperparameters included as default values(Now only support **NUCLA**/**Cross-View**/**Single**/**Gumbel**/**Re-Weighted DYAN**(RHdyan)). \

1. DIR-Dictionary(1145MiB GRAM)

    ```bash    
        python train_DIR_D_NUCLA.py --path_list /dir/to/data  --modelRoot /dir/to/save --bs 64 --lam_f 0.01 --gumbel_thresh 0.5 --gpu_id 0  > /dir/to/save/DIR_D_NUCLA.log
    ```
    Or different loss()
     ```bash    
        python train_DIR_D_NUCLA_lossR_B.py --path_list /dir/to/data  --modelRoot /dir/to/save --bs 64 --lam_f 0.01 --gumbel_thresh 0.5 --gpu_id 0  > /dir/to/save/DIR_D_NUCLA.log
    ```
2. DIR-Classification(1107MiB GRAM),[Pretrained Dictionary](https://northeastern-my.sharepoint.com/:u:/r/personal/luo_dan1_northeastern_edu/Documents/Mine/2021-CrossView/src_code/202410_Journal/pretrained/yuexi_NUCLA_CV_Single_rhDYAN_bi_100.pth?csf=1&web=1&e=4OUaI2) from step 1

    ```bash    
        python train_DIR_cls_noCL_NUCLA.py --path_list /dir/to/data  --modelRoot /dir/to/save --pretrain /path/to/model --bs 64 --lam_f 0.01 --gumbel_thresh 0.5 --gpu_id 0  > /dir/to/save/DIR_cls_noCL_NUCLA.log
    ```
3. DIR-Contrastive Learning(Not Supported Yet)

    ```bash    
        python train_DIR_cls_wiCL_NUCLA.py --path_list /dir/to/data --modelRoot /dir/to/save --pretrain /path/to/model
    ```
4. DIR-Finetune(1023MiB GRAM)(Not Supported Yet)

    ```bash    
        python train_DIR_cls_ft_NUCLA.py --modelRoot /dir/to/save --path_list /dir/to/data --pretrain /path/to/model
    ```

## Human Interaction Recognition experiment can be found: 
    https://github.com/DanLuoNEU/Skeleton-based_Interactive_Action_Recognition_via_Dynamics
    
