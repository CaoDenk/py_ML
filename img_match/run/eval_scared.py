import argparse
import random
import torch
import numpy as np


from torch.utils.data import dataloader
from tqdm import tqdm

from misc import NestedTensor
import torch.nn.functional as F
from loguru import logger

from __init__ import _load_module

_load_module("dataset")
from ScaredDataset import ScaredDataset



def main(args,model,log,epoch):
    # get device
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # data_loader_train, data_loader_val, _ = build_data_loader(args)
    left_dir=r"E:\2019\dataset_3\keyframe_1\data\left_finalpass"
    right_dir=r"E:\2019\dataset_3\keyframe_1\data\right_finalpass"
    disp_dir=r"E:\2019\dataset_3\keyframe_1\data\disparity"

    train_dataset=ScaredDataset(left_dir,right_dir,disp_dir)
    train_loader = dataloader.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)


    disp_mae_loss_sum=0
    three_3px_loss_sum=0
    i=0
    for data in tqdm(train_loader):
        left, right = data['left'].to(device), data['right'].to(device)
        disp, occ_mask, occ_mask_right = data['disp'].to(device), data['occ_mask'].to(device), \
                                        data['occ_mask_right'].to(device)

        bs, _, h, w = left.size()

        # build the input
        inputs = NestedTensor(left, right, disp=disp, occ_mask=occ_mask, occ_mask_right=occ_mask_right)

        # forward pass
        outputs = model(inputs)
        pred_disp=outputs["disp_pred"]
        
        pred_disp.unsqueeze_(0)
        
        disp=data["disp"].to(device)
        mask=disp>0
        
        disp_mae_loss=F.l1_loss(disp[mask],pred_disp[mask]).item()
        
        disp_mae_loss_sum+=disp_mae_loss
        
        abs=torch.abs(disp[mask]-pred_disp[mask])
        count=torch.sum(abs>3)
        three_3px_loss=count/disp[mask].shape[0]
        three_3px_loss_sum+=three_3px_loss
        
       
        i+=1
    
    print(f"avg:disp maeloss {disp_mae_loss_sum/i}  3px {three_3px_loss_sum/i}")
    # build model
    logger.info(f"epoch={epoch} avg:disp maeloss {disp_mae_loss_sum/i}   3px {three_3px_loss_sum/i}")


def myeval_sacred(model,log,epoch,args_):
    # ap = argparse.ArgumentParser('STTR training and evaluation script', parents=[get_args_parser()])
    # args_ = ap.parse_args()
    main(args_,model,log,epoch)

