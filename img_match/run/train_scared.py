import argparse
import random
import torch
import numpy as np

from __init__ import _load_module
_load_module("dataset")

from ScaredDataset import ScaredDataset

_load_module("utilities")
_load_module("module")
# from utilities.train_one_epoch import train_one_epoch
# from train_one_epoch import train_one_epoch



from train_one_epoch import train_one_epoch
from loss import build_criterion
from sttr import STTR

from torch.utils.data import dataloader
from loguru import logger
from eval_scared import myeval_sacred

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
    parser.add_argument('--device', default='cpu',
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

def main(args):
    # get device
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    index=0
    # data_loader_train, data_loader_val, _ = build_data_loader(args)
    left_dir=r"E:\2019\dataset_3\keyframe_2\data\left_finalpass"
    right_dir=r"E:\2019\dataset_3\keyframe_2\data\right_finalpass"
    disp_dir=r"E:\2019\dataset_3\keyframe_2\data\disparity"

    train_dataset=ScaredDataset(left_dir,right_dir,disp_dir)
    train_loader = dataloader.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)
    model = STTR(args).to(device)
    # model=torch.load(rf"sttr_train {index}.pth")

    param_dicts = [
            {"params": [p for n, p in model.named_parameters() if
                        "backbone" not in n and "regression" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": args.lr_backbone,
            },
            {
                "params": [p for n, p in model.named_parameters() if "regression" in n and p.requires_grad],
                "lr": args.lr_regression,
            },
        ]
    # build loss criterion
    criterion = build_criterion(args)
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate)

    # build model

    print("Start training")
    logger.add("eval_scared.log")
    for epoch in range(args.start_epoch, args.epochs):
        # train
        print("Epoch: %d" % epoch)
        train_one_epoch(model, train_loader, optimizer, criterion, device, epoch,
                        args.clip_max_norm, None)

        # step lr if not pretraining
        if not args.pre_train:
            lr_scheduler.step()
            print("current learning rate", lr_scheduler.get_lr())

        # empty cache
        # torch.cuda.empty_cache()
        myeval_sacred(model,logger,epoch=index)
        index +=1
        torch.save(model,f"sttr_train {index}.pth")


    return


if __name__=='__main__':
    ap = argparse.ArgumentParser('STTR training and evaluation script', parents=[get_args_parser()])
    args_ = ap.parse_args()
    main(args_)