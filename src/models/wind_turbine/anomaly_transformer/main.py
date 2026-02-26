import os
import argparse

from torch.backends import cudnn
from utils.utils import *

from solver import Solver


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        solver.test()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--win_size', type=int, default=144)
    #parser.add_argument('--_size', type=int, default=10)
    parser.add_argument('--input_c', type=int, default=82)
    parser.add_argument('--output_c', type=int, default=82)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_', type=str, required=True, choices=['0','1','2','3'], help='anomaly detector model class: 0: generator-bearing 1: gearbox 2: transformer 3: hydraulic')
    parser.add_argument('--dataset', type=str, default='Wind Farm A')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--thresh',type=float, default=None)
    parser.add_argument('--data_path', type=str, default='../datasets')
    parser.add_argument('--model_save_path', type=str, default='model checkpoints')
    parser.add_argument('--background_data_path', type=str, default='.')
    #parser.add_argument('--load_background_data', type=bool, default=False)
    parser.add_argument('--anomalous_inputs_path', type=str, default='.')
    parser.add_argument('--wrapper_path', type=str, default='.')
    #parser.add_argument('--load_anomalous_inputs', type=bool, default=False)
    parser.add_argument('--scaler_path', type=str, default='.')
    #parser.add_argument('--fit', type=bool, default=True)
    parser.add_argument('--discrepancy_table_path', type=str, default='.')
    parser.add_argument('--plot_save_path', type=str, default='plots')
    parser.add_argument('--anormly_ratio', type=float, default=1.5)

    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
