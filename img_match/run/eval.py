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

def get_args_parser():
    """
    Parse arguments
    """

    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-4, type=float)
    parser.add_argument('--lr_regression', default=2e-4, type=float)
    parser.add_argument('--lr_decay_rate', default=0.99, type=float)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--ft', action='store_true', help='load model from checkpoint, but discard optimizer state')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--checkpoint', type=str, default='dev', help='checkpoint name for current experiment')
    parser.add_argument('--pre_train', action='store_true')
    parser.add_argument('--downsample', default=3, type=int,
                        help='This is outdated in STTR-light. Default downsampling is 4 and cannot be changed.')
    parser.add_argument('--apex', action='store_true', help='enable mixed precision training')

    # * STTR
    parser.add_argument('--channel_dim', default=128, type=int,
                        help="Size of the embeddings (dimension of the transformer)")

    # * Positional Encoding
    parser.add_argument('--position_encoding', default='sine1d_rel', type=str, choices=('sine1d_rel', 'none'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--num_attn_layers', default=6, type=int, help="Number of attention layers in the transformer")
    parser.add_argument('--nheads', default=4, type=int,
                        help="Number of attention heads inside the transformer's attentions")

    # * Regression Head
    parser.add_argument('--regression_head', default='ot', type=str, choices=('softmax', 'ot'),
                        help='Normalization to be used')
    parser.add_argument('--context_adjustment_layer', default='cal', choices=['cal', 'none'], type=str)
    parser.add_argument('--cal_num_blocks', default=8, type=int)
    parser.add_argument('--cal_feat_dim', default=16, type=int)
    parser.add_argument('--cal_expansion_ratio', default=4, type=int)

    # * Dataset parameters
    parser.add_argument('--dataset', default='sceneflow', type=str, help='dataset to train/eval on')
    parser.add_argument('--dataset_directory', default='', type=str, help='directory to dataset')
    parser.add_argument('--validation', default='validation', type=str, choices={'validation', 'validation_all'},
                        help='If we validate on all provided training images')

    # * Loss
    parser.add_argument('--px_error_threshold', type=int, default=3,
                        help='Number of pixels for error computation (default 3 px)')
    parser.add_argument('--loss_weight', type=str, default='rr:1.0, l1_raw:1.0, l1:1.0, occ_be:1.0',
                        help='Weight for losses')
    parser.add_argument('--validation_max_disp', type=int, default=-1)

    return parser

def main(args,model,log,epoch):
    # get device
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # data_loader_train, data_loader_val, _ = build_data_loader(args)
    left_dir=r"E:\Dataset\endo_depth\22_crop_288_496\val\image01"
    right_dir=r"E:\Dataset\endo_depth\22_crop_288_496\val\image02"
    disp_dir=r"E:\Dataset\endo_depth\22_crop_288_496\val\disp_np01"

    train_dataset=ScaredDataset(left_dir,right_dir,disp_dir)
    train_loader = dataloader.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)
    # model=torch.load(r"E:\中期\sttr_light\stereo-transformer\sttr_train 177.pth")
    
    fx=417.9036255
    baseline=5.2864
    fb=fx*baseline
    
    depth_mae_loss_sum=0
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
        
        
        depth=fb/disp[mask]
        pred_depth=fb/pred_disp[mask]
        
        depth_mae_loss=F.l1_loss(depth,pred_depth).item()
        
        depth_mae_loss_sum+=depth_mae_loss
        
        abs=torch.abs(disp[mask]-pred_disp[mask])
        
        three_3px_loss=torch.sum(abs>3)/disp[mask].shape[0]
        three_3px_loss_sum+=three_3px_loss
        # print(f"disp mae{disp_mae_loss}  depth mae loss {depth_mae_loss} 3px {three_3px_loss}")
        # break
        i+=1
    
    print(f"avg:disp maeloss {disp_mae_loss_sum/i}  depth mae loss {depth_mae_loss_sum/i} 3px {three_3px_loss_sum/i}")
    # build model
    logger.info(f"epoch={epoch} avg:disp maeloss {disp_mae_loss_sum/i}  depth mae loss {depth_mae_loss_sum/i} 3px {three_3px_loss_sum/i}")


def myeval(model,log,epoch):
    ap = argparse.ArgumentParser('STTR training and evaluation script', parents=[get_args_parser()])
    args_ = ap.parse_args()
    main(args_,model,log,epoch)

if __name__=='__main__':
    ap = argparse.ArgumentParser('STTR training and evaluation script', parents=[get_args_parser()])
    args_ = ap.parse_args()
    main(args_)