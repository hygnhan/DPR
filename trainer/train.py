import numpy as np
import torch as t
import os
import argparse
import random

from learner import Learner
from module.logger import *

from analysis import Analysis

t.backends.cudnn.deterministic = True
t.backends.cudnn.benchmark = False



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mitigating Spurious Correlations via Disagreement Probability')

    parser.add_argument('--model', help='model', default='MLP', type=str)
    parser.add_argument('--batch_size', help='batch size', default=256, type=int)
    parser.add_argument('--lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument('--lr_decay', help='learning rate decay', default=0.9, type=float)
    parser.add_argument('--weight_decay', help='weight decay', default=0.0, type=float)
    parser.add_argument('--momentum', help='momentum number', default=0.9, type=float)

    parser.add_argument('--seed', help='random seed', default=0, type=int)
    parser.add_argument('--num_workers', help='workers number', default=4, type=int)
    parser.add_argument('--exp', help='experiment name', default='debugging', type=str)
    parser.add_argument('--device', help='cuda or cpu', default='cuda', type=str)
    parser.add_argument('--epochs', help='# of epochs', default=100, type=int)
    parser.add_argument('--n_epochs', help='# of epochs', default=100, type=int)
    parser.add_argument('--dataset', help='dataset', default= 'colored_mnist', type=str)
    parser.add_argument('--bratio', help='minorty percentage', default= 0.01, type=float)
    parser.add_argument('--use_lr_decay', action='store_true', help='whether to use learning rate decay')
    parser.add_argument('--lr_decay_step', help='learning rate decay steps', type=int, default=10000)
    parser.add_argument('--norm_scale', help='Norm scale', type=float, default=2)
    parser.add_argument('--algorithm',  help='run algorithm', default='vanilla',    type=str)

    parser.add_argument('--log_dir', help='path for saving model', default='./log/', type=str)
    parser.add_argument('--data_dir', help='path for loading data', default='./dataset/', type=str)
    
    parser.add_argument('--reproduce',  help='Reproduce',       action='store_true')
    parser.add_argument('--pretrained_bmodel',  help='Use pretrained biased model?',       action='store_true')
    parser.add_argument('--pretrained_dmodel',  help='Use pretrained debiased model?',       action='store_true')
    parser.add_argument('--train',              help='Train?',       action='store_true')
    parser.add_argument('--save_stats',         help='Save stats?',       action='store_true')

    # experiments
    parser.add_argument('--opt', help='SGD, Adam', type=str)
    parser.add_argument('--num_class', help='# of classes', type=int)
    parser.add_argument('--save_sampling_prob',    action='store_true')
    parser.add_argument('--tau',  help='temperature', default=1, type=float)
    parser.add_argument('--gamma',  help='hyperparameter for exp2',    type=float)
    parser.add_argument('--scheduler',  type=str)
    parser.add_argument('--scheduler_param', default=100, type=int)

    # save the best model using validation accuracy criteria
    parser.add_argument('--val_acc',    action='store_true')
    parser.add_argument('--method', default=None, type=str)
    parser.add_argument('--aug', default=None, type=str)
    parser.add_argument('--bias', default=None, type=str, choices=[None, 'aligned', 'conflict'])

    # hyperparameters of LC
    parser.add_argument("--curr_step", help="curriculum steps", type=int, default= 2)
    parser.add_argument("--lambda_dis_align",  help="lambda_dis in Eq.2", type=float, default=1.0)
    parser.add_argument("--avg_type", help="pya estimation types", type=str, default='mv')
    parser.add_argument("--ema_alpha",  help="use weight mul", type=float, default=0.5)

    parser.add_argument('--q', help='GCE parameter q', type=float, default=0.7)

    args = parser.parse_args()

    args.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
    
    # Path
    args.exp = args.exp+str(args.seed)
    args.log_dir = args.log_dir+args.dataset+'/'+args.algorithm+'/'+'bias_'+str(args.bratio)+'/'+args.exp+'/'
    os.makedirs(args.log_dir, exist_ok=True)
    
    
    # Logger
    _logger = logger(args)
    args.print = _logger.critical
    args.write = _logger.debug

    # Reproducibility
    if args.reproduce:
        from utils.pre_conf import *
        args = reproduce(args)
    
    # Random seed
    t.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    t.cuda.manual_seed(args.seed)
    t.cuda.manual_seed_all(args.seed)

    # Logging current configuration
    args.print(args)

    # Trainer setting
    learner = Learner(args)

    # train
    if args.train:
        if args.algorithm == 'vanilla':
            learner.train_vanilla()
        
        elif args.algorithm == 'lff':
            learner.train_lff()

        elif args.algorithm == 'rebias':
            learner.train_rebias()

        elif args.algorithm == 'dfa':
            learner.train_dfa()

        elif args.algorithm == 'jtt':
            learner.train_jtt()

        elif args.algorithm == 'lc':
            learner.train_lc()
            
        elif args.algorithm == 'pgd':
            learner.train_pgd()
        
        elif args.algorithm == 'dpr':
            learner.train_dpr()

        else:
            print('Wrong algorithm')
            import sys
            sys.exit(0)


    else:
        if args.algorithm == 'vanilla':
            learner.evaluate(bias=True)
        
        elif args.algorithm == 'lff':
            learner.evaluate(debias=True)

        elif args.algorithm == 'rebias':
            learner.evaluate(debias=True)

        elif args.algorithm == 'dfa':
            learner.evaluate(bias=True, debias=True)

        elif args.algorithm == 'jtt':
            learner.evaluate(debias=True)
        
        elif args.algorithm == 'lc':
            learner.evaluate(debias=True)

        elif args.algorithm == 'pgd':
            learner.evaluate(debias=True)    

        elif args.algorithm == 'dpr':
            learner.evaluate(debias=True)

        else:
            print('Wrong algorithm')
            import sys
            sys.exit(0)
