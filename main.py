
# -*- coding: utf-8 -*- 


import argparse
from train import *

## Parser
parser = argparse.ArgumentParser(description='Training models', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=0.0001, type=float, dest="lr")
parser.add_argument("--batch_size", default=128, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=10, type=int, dest="num_epoch")
parser.add_argument("--load_opt", default=None, type=int, dest="load_opt")
parser.add_argument("--data_dir", default='../gender_738k/', type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default='../checkpoint', type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default='../log', type=str, dest="log_dir")
parser.add_argument("--result_dir", default="../result", type=str, dest="result_dir")
parser.add_argument("--date", default=None, type=str, dest="date")
parser.add_argument('--mode', default='train', choices=['train', 'test'], dest='mode')
parser.add_argument('--train_continue', default='off', choices=['on', 'off'], dest='train_continue')

#parser.add_argument("--task", default="resnet", choices=["resnet", "mobilenet", "efficientnet"], type=str, dest="task")

parser.add_argument('--ny', type=int, default=112, dest='ny')
parser.add_argument('--nx', type=int, default=112, dest='nx')
parser.add_argument('--nch', type=int, default=3, dest='nch')
#parser.add_argument('--nker', type=int, default=128, dest='nker')

parser.add_argument("--network", default = "mobilenet", choices=["resnet", "mobilenet", "efficientnet"], type=str, dest="network")


args = parser.parse_args()


if __name__ == "__main__":
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
