from tqdm import tqdm
import numpy as np
import torch as t
import torch.nn as nn

import os
import pickle as pkl
import torch.optim as optim
from torch.utils.data import DataLoader

from module.loss import GeneralizedCELoss, LogitCorrectionLoss
from module.net import *
from module.data import *
from module.avgmeter import *
from module.ema import *
from module.gradient import *
from module.scores import *

from utils.lc_utils import sigmoid_rampup, group_mixUp, EMA, EMA_squre
from utils.analysis import *
import time
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        
class Learner(object):
    def __init__(self,args):
        self.device = t.device(args.device)
        self.args = args

        if args.algorithm == 'lc' and args.dataset == 'bffhq':
            args.batch_size = 256
        
        train_dataset=get_dataset(  dataset=args.dataset,
                                    data_dir=args.data_dir,
                                    split='train',
                                    bias=args.bratio,
                                    args=args)
    
        valid_dataset=get_dataset(  dataset=args.dataset,
                                    data_dir=args.data_dir,
                                    split='valid',
                                    bias=args.bratio,
                                    args=args)
    
        test_dataset=get_dataset(   dataset=args.dataset,
                                    data_dir=args.data_dir,
                                    split='test',
                                    bias=args.bratio,
                                    args=args)

        self.train_loader = DataLoader( train_dataset,
                                        batch_size = args.batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers,
                                        pin_memory=True,
                                        drop_last=False)

        self.valid_loader = DataLoader( valid_dataset,
                                        batch_size = args.batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers,
                                        pin_memory=True,
                                        drop_last=False)
        
        self.test_loader = DataLoader(  test_dataset,
                                        batch_size = args.batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers,
                                        pin_memory=True,
                                        drop_last=False)
        
        self.model_b = get_model(args.model, args.num_class).to(self.device)
        self.model_d = get_model(args.model, args.num_class).to(self.device)
        
        
        if args.opt == 'SGD':
            self.optimizer_b = optim.SGD(
                self.model_b.parameters(),
                lr = args.lr,
                weight_decay = args.weight_decay,
                momentum = args.momentum
            )

            self.optimizer_d = optim.SGD(
                self.model_d.parameters(),
                lr = args.lr,
                weight_decay = args.weight_decay,
                momentum = args.momentum
            )

        elif args.opt == 'Adam':
            self.optimizer_b = optim.Adam(
                self.model_b.parameters(),
                lr = args.lr,
                weight_decay = args.weight_decay
            )


            self.optimizer_d = optim.Adam(
                self.model_d.parameters(),
                lr = args.lr,
                weight_decay = args.weight_decay
            )
        
        self.step_b = optim.lr_scheduler.StepLR(self.optimizer_b,step_size = self.args.lr_decay_step, gamma=self.args.lr_decay)
        self.step_d = optim.lr_scheduler.StepLR(self.optimizer_d,step_size = self.args.lr_decay_step, gamma=self.args.lr_decay)

        self.loss_b = nn.CrossEntropyLoss(reduction='none')
        self.loss_d = nn.CrossEntropyLoss(reduction='none')

        self.best_loss = np.inf
        self.best_val_acc = 0
        self.best_acc = 0
        self.best_epoch = 0

    def reset_meter(self,epoch):
        self.epoch = epoch

        self.b_train_acc    = AvgMeter()
        self.b_train_loss   = AvgMeter()
        self.d_train_acc    = AvgMeter()
        self.d_train_loss   = AvgMeter()
        
        
        self.val_acc      = AvgMeter()
        self.val_loss     = AvgMeter()

        self.test_acc     = AvgMeter()
        self.test_loss    = AvgMeter()


    def save_models(self,option,debias=False,bias=False):
        if 'ResNet20' in self.args.model:
            if debias:
                t.save(self.model_d.state_dict(), self.args.log_dir+'model_d_'+option+'.pt')
            if bias:
                t.save(self.model_b.state_dict(), self.args.log_dir+'model_b_'+option+'.pt')
        
        else:
            if debias:
                t.save(self.model_d, self.args.log_dir+'model_d_'+option+'.pt')
            if bias:
                t.save(self.model_b, self.args.log_dir+'model_b_'+option+'.pt')
        

    def load_models(self,option,debias=False,bias=False):
        print('=====')
        if self.args.model == 'ResNet20':
            if debias:
                self.model_d.load_state_dict(t.load(self.args.log_dir+'model_d_'+option+'.pt'))
                self.model_d = self.model_d.to(self.device)
                print(f'load debiased model: {self.args.log_dir}model_d_{option}.pt')
            if bias:
                self.model_b.load_state_dict(t.load(self.args.log_dir+'model_b_'+option+'.pt'))
                self.model_b = self.model_b.to(self.device)
                print(f'load biased model: {self.args.log_dir}model_b_{option}.pt')
        
        else:
            if debias:
                self.model_d = t.load(self.args.log_dir+'model_d_'+option+'.pt').to(self.device)
                print(f'load debiased model: {self.args.log_dir}model_d_{option}.pt')
            if bias:
                self.model_b = t.load(self.args.log_dir+'model_b_'+option+'.pt').to(self.device)
                print(f'load biased model: {self.args.log_dir}model_b_{option}.pt')
        
        print('=====')


    def evaluate(self,debias=False,bias=False):
        self.epoch = -100
        self.load_models('end',debias=debias,bias=bias)
        self.test_acc = AvgMeter()
        self.test_loss = AvgMeter()
        self.test(debias = debias, bias = bias)
        self.args.print('End Los %.4f' %(self.test_loss.avg))
        self.args.print('End Acc %.4f' %(self.test_acc.avg))
        print('End Los %.4f' %(self.test_loss.avg))
        print('End Acc %.4f' %(self.test_acc.avg))

        self.load_models('best',debias=debias,bias=bias)
        self.test_acc = AvgMeter()
        self.test_loss = AvgMeter()
        self.test(debias = debias, bias = bias)
        self.args.print('Best Los %.4f' %(self.test_loss.avg))
        self.args.print('Best Acc %.4f' %(self.test_acc.avg))
        print('Best Los %.4f' %(self.test_loss.avg))
        print('Best Acc %.4f' %(self.test_acc.avg))


    def print_result(self, type='both'):
        self.args.print('Epoch - %3d / %3d' %(self.epoch+1, self.args.epochs))
        if self.args.val_acc:
            self.args.print('Valid Best acc - (Epoch: %3d) %.4f // %.4f' %(self.best_epoch+1, self.best_val_acc, self.best_acc))
        else:
            self.args.print('Valid Best loss - (Epoch: %3d) %.4f // %.4f' %(self.best_epoch+1, self.best_loss, self.best_acc))
        
        if type == 'debias':
            self.args.print('Debiased Train Los %.4f' %(self.d_train_loss.avg))
            self.args.print('Debiased Train Acc %.4f' %(self.d_train_acc.avg))
        elif type == 'bias':
            self.args.print('Biased Train Los %.4f' %(self.b_train_loss.avg))
            self.args.print('Biased Train Acc %.4f' %(self.b_train_acc.avg))
        else:
            self.args.print('Debiased Train Los %.4f' %(self.d_train_loss.avg))
            self.args.print('Debiased Train Acc %.4f' %(self.d_train_acc.avg))

            self.args.print('Biased Train Los %.4f' %(self.b_train_loss.avg))
            self.args.print('Biased Train Acc %.4f' %(self.b_train_acc.avg))
        
    

        self.args.print('Valid Los %.4f' %(self.val_loss.avg))
        self.args.print('Valid Acc %.4f' %(self.val_acc.avg))

        self.args.print('Test Los %.4f' %(self.test_loss.avg))
        self.args.print('Test Acc %.4f' %(self.test_acc.avg))
        
        

    def test(self, debias = False, bias=False):
        self.model_b.eval()
        self.model_d.eval()
        
        for _, data_tuple in enumerate(self.test_loader):
            data = data_tuple[0].to(self.device)
            label = data_tuple[1].to(self.device)
            bias_label = data_tuple[2]
            idx = data_tuple[-1]

            if bias and debias:
                feature_d = self.model_d.extract(data)
                feature_b = self.model_b.extract(data)

                feature = t.cat((feature_d,feature_b),dim=1)

                logit_d = self.model_d.predict(feature)
                loss_d = self.loss_d(logit_d,label)

                loss = loss_d.mean().float().detach().cpu()
                acc = t.mean((logit_d.max(1)[1] == label).float().detach().cpu())
            else:
                if debias:
                    logit_d = self.model_d(data)
                    loss_d = self.loss_d(logit_d, label)
                    loss = loss_d.mean().float().detach().cpu()
                    acc = t.mean((logit_d.max(1)[1] == label).float().detach().cpu())

                else:            
                    logit_b = self.model_b(data)
                    loss_b = self.loss_b(logit_b, label)
                    loss = loss_b.mean().float().detach().cpu()
                    acc = t.mean((logit_b.max(1)[1] == label).float().detach().cpu())
            
            self.test_loss.update(loss,len(label))
            self.test_acc.update(acc, len(label))
        
        if self.epoch == self.best_epoch:
            self.best_acc = self.test_acc.avg

    def validate(self, debias = False, bias = False):
        self.model_b.eval()
        self.model_d.eval()
        
        for _, data_tuple in enumerate(self.valid_loader):
            data = data_tuple[0].to(self.device)
            label = data_tuple[1].to(self.device)
            bias_label = data_tuple[2]
            idx = data_tuple[-1]
            
            if bias and debias:
                feature_d = self.model_d.extract(data)
                feature_b = self.model_b.extract(data)

                feature = t.cat((feature_d,feature_b),dim=1)

                logit_d = self.model_d.predict(feature)
                loss_d = self.loss_d(logit_d,label)

                loss = loss_d.mean().float().detach().cpu()
                acc = t.mean((logit_d.max(1)[1] == label).float().detach().cpu())
                
            else:
                if debias:
                    logit_d = self.model_d(data)
                    loss_d = self.loss_d(logit_d, label)
                    loss = loss_d.mean().float().detach().cpu()

                    acc = t.mean((logit_d.max(1)[1] == label).float().detach().cpu())
                    
                    
                else:
                    logit_b = self.model_b(data)
                    loss_b = self.loss_b(logit_b, label)
                    loss = loss_b.mean().float().detach().cpu()

                    acc = t.mean((logit_b.max(1)[1] == label).float().detach().cpu())
                    
            self.val_loss.update(loss,len(label))
            self.val_acc.update(acc, len(label))

        if self.args.val_acc:
            if self.val_acc.avg >= self.best_val_acc:
                self.best_epoch = self.epoch
                self.best_val_acc = self.val_acc.avg
                self.save_models('best',debias = debias, bias = bias)
        else:
            if self.val_loss.avg <= self.best_loss:
                self.best_epoch = self.epoch
                self.best_loss = self.val_loss.avg
                self.save_models('best',debias = debias, bias = bias)

    def train_vanilla(self):
        self.best_loss = np.inf
        self.best_val_acc = 0
        self.best_acc = 0
        for epoch in range(self.args.epochs):
            self.model_b.train()
            self.model_d.train()
            self.reset_meter(epoch)
            for _, data_tuple in enumerate(self.train_loader):
                data = data_tuple[0].to(self.device)
                label = data_tuple[1].to(self.device)
                bias_label = data_tuple[2]
                idx = data_tuple[-1]

                logit_b = self.model_b(data)
                loss_b = self.loss_b(logit_b, label)
                loss = loss_b.mean()

                self.optimizer_b.zero_grad()
                loss.backward()
                self.optimizer_b.step()

                acc = t.mean((logit_b.max(1)[1] == label).float().detach().cpu())

                self.b_train_loss.update(loss,len(label))
                self.b_train_acc.update(acc, len(label))
            self.validate(bias = True)
            self.test(bias=True)
            self.print_result(type='bias')
            self.step_b.step()
        self.save_models('end', bias=True)

    def train_rebias(self):
        from module.hsic import MinusRbfHSIC, RbfHSIC
        
        b_HSIC = MinusRbfHSIC()
        d_HSIC = RbfHSIC()

        self.best_loss = np.inf
        self.best_val_acc = 0
        self.best_acc = 0
        for epoch in range(self.args.epochs):
            self.model_b.train()
            self.model_d.train()
            self.reset_meter(epoch)
            for _, data_tuple in tqdm(enumerate(self.train_loader)):
                data = data_tuple[0].to(self.device)
                label = data_tuple[1].to(self.device)
                bias_label = data_tuple[2]
                idx = data_tuple[-1]

                feature_b = self.model_b.extract(data)
                logit_b = self.model_b.predict(feature_b)
                feature_d = self.model_d.extract(data)
                if len(label)>2:
                    loss_b = self.loss_b(logit_b, label) + b_HSIC(feature_b, feature_d)
                else:
                    loss_b = self.loss_b(logit_b, label)
                loss = loss_b.mean()
                acc = t.mean((logit_b.max(1)[1] == label).float().detach().cpu())
                self.optimizer_b.zero_grad()
                loss.backward()
                self.optimizer_b.step()
                self.b_train_loss.update(loss,len(label))
                self.b_train_acc.update(acc,len(label))

                feature_d = self.model_d.extract(data)
                logit_d = self.model_d.predict(feature_d)
                feature_b = self.model_b.extract(data)
                

                if len(label) > 2:                
                    loss_d = self.loss_d(logit_d, label) + d_HSIC(feature_d, feature_b)
                else:
                    loss_d = self.loss_d(logit_d, label)
                loss = loss_d.mean()
                acc = t.mean((logit_d.max(1)[1] == label).float().detach().cpu())
                self.optimizer_d.zero_grad()
                loss.backward()
                self.optimizer_d.step()
                self.d_train_loss.update(loss,len(label))
                self.d_train_acc.update(acc,len(label))

            self.validate(debias=True)
            self.test(debias=True)
            self.print_result(type='both')
            self.step_b.step()
            self.step_d.step()
        self.save_models('end', bias=True, debias=True)

    def train_lff(self):
        # Initialize
        self.best_loss = np.inf
        self.best_val_acc = 0
        self.best_acc = 0
        self.sample_loss_ema_b = EMA(t.LongTensor(self.train_loader.dataset.label).to(self.device), alpha=0.7)
        self.sample_loss_ema_d = EMA(t.LongTensor(self.train_loader.dataset.label).to(self.device), alpha=0.7)
        self.bias_criterion = GeneralizedCELoss(q=0.7)

        for epoch in range(self.args.epochs):
            self.model_b.train()
            self.model_d.train()
            self.reset_meter(epoch)
            for _, data_tuple in tqdm(enumerate(self.train_loader)):
                data = data_tuple[0].to(self.device)
                label = data_tuple[1].to(self.device)
                bias_label = data_tuple[2]
                idx = data_tuple[-1]

                logit_b = self.model_b(data)
                logit_d = self.model_d(data)
                
                loss_b = self.loss_b(logit_b,label).detach()
                loss_d = self.loss_d(logit_d,label).detach()

                _loss_b = loss_b.mean()
                _loss_d = loss_d.mean()

                self.sample_loss_ema_b.update(loss_b, idx.to(self.device))
                self.sample_loss_ema_d.update(loss_d, idx.to(self.device))

                loss_b = self.sample_loss_ema_b.parameter[idx].clone().detach()
                loss_d = self.sample_loss_ema_d.parameter[idx].clone().detach()

                label_cpu = label.cpu()

                for c in range(self.args.num_class):
                    class_idx = t.where(label_cpu == c)[0]
                    max_loss_b = self.sample_loss_ema_b.max_loss(c)
                    max_loss_d = self.sample_loss_ema_d.max_loss(c)
                    loss_b[class_idx] /= max_loss_b
                    loss_d[class_idx] /= max_loss_d
                
                loss_weight = loss_b / (loss_b + loss_d + 1e-8)
                loss_b_update = self.bias_criterion(logit_b, label)
                loss_d_update = self.loss_d(logit_d, label) * loss_weight.to(self.device)
                loss = loss_b_update.mean() + loss_d_update.mean()
                
                self.optimizer_b.zero_grad()
                self.optimizer_d.zero_grad()
                loss.backward()
                self.optimizer_b.step()
                self.optimizer_d.step()

                _acc_b = t.mean((logit_b.max(1)[1] == label).float().detach().cpu())
                _acc_d = t.mean((logit_d.max(1)[1] == label).float().detach().cpu())

                self.d_train_loss.update(_loss_d,len(label))
                self.b_train_loss.update(_loss_b,len(label))
                self.d_train_acc.update(_acc_d,len(label))
                self.b_train_acc.update(_acc_b,len(label))
            self.validate(debias=True)
            self.test(debias=True)
            self.print_result(type='both')
            self.step_b.step()
            self.step_d.step()
        self.save_models('end', bias=True, debias=True)

    def train_dfa(self):
        # Initialize
        self.args.model += '_DIS'
        self.model_b = get_model(self.args.model, self.args.num_class).to(self.device)
        self.model_d = get_model(self.args.model, self.args.num_class).to(self.device)

        if self.args.opt == 'Adam':
            self.optimizer_b = optim.Adam(
                self.model_b.parameters(),
                lr = self.args.lr,
                weight_decay = self.args.weight_decay
            )
            self.optimizer_d = optim.Adam(
                self.model_d.parameters(),
                lr = self.args.lr,
                weight_decay = self.args.weight_decay
            )
        elif self.args.opt == 'SGD':
            self.optimizer_b = optim.SGD(
                self.model_b.parameters(),
                lr = self.args.lr,
                weight_decay = self.args.weight_decay,
                momentum = self.args.momentum
            )
            self.optimizer_d = optim.SGD(
                self.model_d.parameters(),
                lr = self.args.lr,
                weight_decay = self.args.weight_decay,
                momentum = self.args.momentum
            )

        self.step_b = optim.lr_scheduler.StepLR(self.optimizer_b,step_size = self.args.lr_decay_step, gamma=self.args.lr_decay)
        self.step_d = optim.lr_scheduler.StepLR(self.optimizer_d,step_size = self.args.lr_decay_step, gamma=self.args.lr_decay)

        self.best_loss = np.inf
        self.best_val_acc = 0
        self.best_acc = 0
        self.sample_loss_ema_b = EMA(t.LongTensor(self.train_loader.dataset.label).to(self.device), alpha=0.7)
        self.sample_loss_ema_d = EMA(t.LongTensor(self.train_loader.dataset.label).to(self.device), alpha=0.7)
        self.bias_criterion = GeneralizedCELoss(q=0.7)
        lambda_dis = 10
        lambda_swap = 10
        lambda_swap_b = 1
        curr_step = 50


        for epoch in range(self.args.epochs):
            self.model_b.train()
            self.model_d.train()
            self.reset_meter(epoch)
            for _, data_tuple in enumerate(self.train_loader):
                data = data_tuple[0].to(self.device)
                label = data_tuple[1].to(self.device)
                bias_label = data_tuple[2]
                idx = data_tuple[-1]


                _z_d = self.model_d.extract(data)
                _z_b = self.model_b.extract(data)

                z_d = t.cat((_z_d,_z_b.detach()),dim=1)
                z_b = t.cat((_z_d.detach(),_z_b),dim=1)

                pred_d = self.model_d.predict(z_d)
                pred_b = self.model_b.predict(z_b)

                loss_d = self.loss_d(pred_d, label).detach()
                loss_b = self.loss_b(pred_b,label).detach()

                _loss_d = loss_d.mean()
                _loss_b = loss_b.mean()

                self.sample_loss_ema_d.update(loss_d,idx.to(self.device))
                self.sample_loss_ema_b.update(loss_b,idx.to(self.device))

                loss_b = self.sample_loss_ema_b.parameter[idx].clone().detach()
                loss_d = self.sample_loss_ema_d.parameter[idx].clone().detach()
                
                label_cpu = label.cpu()

                for c in range(self.args.num_class):
                    class_idx = t.where(label_cpu == c)[0]
                    max_loss_b = self.sample_loss_ema_b.max_loss(c)
                    max_loss_d = self.sample_loss_ema_d.max_loss(c)
                    loss_b[class_idx] /= max_loss_b
                    loss_d[class_idx] /= max_loss_d
                
                loss_weight = loss_b / (loss_b + loss_d + 1e-8)
                loss_b_update = self.bias_criterion(pred_b, label)
                loss_d_update = self.loss_d(pred_d, label) * loss_weight.to(self.device)
                
                if epoch > curr_step:
                    indices = np.random.permutation(_z_b.size(0))
                    _z_b_swap = _z_b[indices]
                    label_swap = label[indices]

                    z_mix_d = t.cat((_z_d,_z_b_swap.detach()),dim=1)
                    z_mix_b = t.cat((_z_d.detach(),_z_b_swap),dim=1)

                    pred_mix_d = self.model_d.predict(z_mix_d)
                    pred_mix_b = self.model_b.predict(z_mix_b)

                    loss_swap_d_update = self.loss_d(pred_mix_d,label) * loss_weight.to(self.device)
                    loss_swap_b_update = self.bias_criterion(pred_mix_b, label_swap)

                else:
                    loss_swap_d_update = t.tensor([0]).float()
                    loss_swap_b_update = t.tensor([0]).float()

                loss_dis = loss_b_update.mean() + loss_d_update.mean() * lambda_dis
                loss_swap = loss_swap_b_update.mean() + loss_swap_d_update.mean() * lambda_swap
                loss = loss_dis + loss_swap
                
                self.optimizer_b.zero_grad()
                self.optimizer_d.zero_grad()
                loss.backward()
                self.optimizer_b.step()
                self.optimizer_d.step()

                _acc_b = t.mean((pred_b.max(1)[1] == label).float().detach().cpu())
                _acc_d = t.mean((pred_d.max(1)[1] == label).float().detach().cpu())

                self.d_train_loss.update(_loss_d,len(label))
                self.b_train_loss.update(_loss_b,len(label))
                self.d_train_acc.update(_acc_d,len(label))
                self.b_train_acc.update(_acc_b,len(label))

                del(data)
                del(label)

            self.validate(bias=True, debias=True)
            self.test(bias=True, debias=True)
            self.print_result(type='both')
            self.step_b.step()
            self.step_d.step()
        self.save_models('end',bias=True, debias=True)

    def train_jtt(self):
        # Initialize
        self.best_loss = np.inf
        self.best_val_acc = 0
        self.best_acc = 0

        # Bias Train
        if not self.args.pretrained_bmodel:
            for epoch in range(10):
                self.model_b.train()
                self.model_d.train()
                self.reset_meter(epoch)
                for _, data_tuple in enumerate(self.train_loader):
                    data = data_tuple[0].to(self.device)
                    label = data_tuple[1].to(self.device)
                    gt_label = data_tuple[2]
                    bias_label = data_tuple[3]
                    idx = data_tuple[-1]

                    logit_b = self.model_b(data)
                    loss_b = self.loss_b(logit_b, label)
                    loss = loss_b.mean()

                    self.optimizer_b.zero_grad()
                    loss.backward()
                    self.optimizer_b.step()

                    acc = t.mean((logit_b.max(1)[1] == label).float().detach().cpu())

                    self.b_train_loss.update(loss,len(label))
                    self.b_train_acc.update(acc, len(label))
                self.validate(bias = True)
                self.test(bias = True)
                self.print_result(type='bias')
                self.step_b.step()
            self.save_models('end',bias=True)
        self.load_models('end',bias=True)

        # Calcuate probability
        self.model_b.eval()
        
        output = t.zeros(len(self.train_loader.dataset)).int()

        for bidx, data_tuple in enumerate(self.train_loader):
            data = data_tuple[0].to(self.device).requires_grad_(True)
            label = data_tuple[1].to(self.device)
            idx = data_tuple[-1]

            logit_b = self.model_b(data)
            acc = (logit_b.max(1)[1] != label).int()
            output[idx] = acc.detach().cpu()
        
        # lbd_up = 10
        # lbd_up = (1-self.args.bratio)/ self.args.bratio
        lbd_up = 30
        output = (output * lbd_up + 1).float()
        output = output / t.sum(output)
        self.train_loader.dataset.update_prob(output)
        self.train_loader.dataset.prob_sample_on()
        
        # self.model_d.load_state_dict(self.model_b.state_dict())
        if not self.args.pretrained_dmodel:
            # Re-initialize
            self.best_loss = np.inf
            self.best_val_acc = 0
            self.best_acc = 0
            # Debiased train
            for epoch in range(self.args.epochs):
                self.model_b.train()
                self.model_d.train()
                self.reset_meter(epoch)
                for _, data_tuple in enumerate(self.train_loader):
                    data = data_tuple[0].to(self.device)
                    label = data_tuple[1].to(self.device)
                    gt_label = data_tuple[2]
                    bias_label = data_tuple[3]
                    idx = data_tuple[-1]

                    logit_d = self.model_d(data)
                    loss_d = self.loss_d(logit_d, label)
                    loss = loss_d.mean()

                    self.optimizer_d.zero_grad()
                    loss.backward()
                    self.optimizer_d.step()

                    acc = t.mean((logit_d.max(1)[1] == label).float().detach().cpu())

                    self.d_train_loss.update(loss,len(label))
                    self.d_train_acc.update(acc, len(label))
                self.validate(debias=True)
                self.test(debias=True)
                self.print_result(type='debias')
                self.step_d.step()
            self.save_models('end',debias=True)
        self.load_models('best',debias=True)

    def train_lc(self):
        epoch, cnt = 0, 0
        self.sample_margin_ema_b = EMA(t.LongTensor(self.train_loader.dataset.label), num_classes=self.args.num_class, alpha=0)
        self.confusion = EMA_squre(num_classes=self.args.num_class, alpha=self.args.ema_alpha, avg_type = self.args.avg_type)

        self.bias_criterion = GeneralizedCELoss(q=self.args.q)
        self.criterion = LogitCorrectionLoss(eta = 1.0)

        step = 0
        for epoch in range(self.args.epochs):
            self.model_b.train()
            self.model_d.train()
            self.reset_meter(epoch)
            for _, data_tuple in enumerate(self.train_loader):
                data = data_tuple[0].to(self.device)
                label = data_tuple[1].to(self.device)
                bias_label = data_tuple[2]
                idx = data_tuple[-1]

                alpha = sigmoid_rampup(epoch, self.args.curr_step)*0.5

                z_b = self.model_b.extract(data)
                z_b = z_b.view(z_b.size(0),-1)
                pred_align = self.model_b.fc(z_b)
                self.sample_margin_ema_b.update(F.softmax(pred_align.detach()/self.args.tau), idx)
                pred_align_mv = self.sample_margin_ema_b.parameter[idx].clone().detach()
                _, pseudo_label = t.max(pred_align_mv, dim=1)
                self.confusion.update(pred_align_mv, label, pseudo_label, fix = None)
                correction_matrix = self.confusion.parameter.clone().detach()
                if self.args.avg_type == 'epoch':
                    correction_matrix = correction_matrix/self.confusion.global_count_.to(self.device)

                correction_delta = correction_matrix[:,pseudo_label]
                correction_delta = t.t(correction_delta)
                return_dict = group_mixUp(data, pseudo_label, correction_delta, label, self.args.num_class, alpha)
                mixed_target_data = return_dict["mixed_feature"]
                mixed_biased_prediction = return_dict["mixed_correction"]
                label_a = return_dict["label_majority"]
                label_b = return_dict["label_minority"]
                lam_target = return_dict["lam"]

                z_d = self.model_d.extract(mixed_target_data)
                z_d = z_d.view(z_d.size(0),-1)
                pred_conflict = self.model_d.fc(z_d)

                self.sample_margin_ema_b.update(F.softmax(pred_align.detach()), idx)               

                loss_dis_conflict = lam_target * self.criterion(pred_conflict, label_a, mixed_biased_prediction) +\
                 (1 - lam_target) * self.criterion(pred_conflict, label_b, mixed_biased_prediction)
         
                loss_dis_align = self.bias_criterion(pred_align, label) 
                loss  = loss_dis_conflict.mean() + self.args.lambda_dis_align * loss_dis_align.mean()               # Eq.2 L_dis
                
                self.optimizer_d.zero_grad()
                self.optimizer_b.zero_grad()

                loss.backward()
                
                # self.scheduler_b.step()
                # self.scheduler_d.step()
                
                self.optimizer_d.step()
                self.optimizer_b.step()

                self.d_train_loss.update(loss,len(label))
                
                step += 1

            confusion_numpy = correction_matrix.cpu().numpy()

            self.confusion.global_count_ = t.zeros(self.args.num_class, self.args.num_class)
            if self.args.avg_type == 'epoch':
                self.confusion.initiate_parameter()
            
            self.validate(debias=True)
            self.test(debias=True)
            self.print_result(type='debias')
            self.step_b.step()
            self.step_d.step()
        self.save_models('end',bias=True)
        self.save_models('end',debias=True)
        
    def train_pgd(self):
        # Initialize
        self.bias_criterion = GeneralizedCELoss(q=0.7)
        self.best_loss = np.inf
        self.best_acc = 0

        # Bias Train
        if not self.args.pretrained_bmodel:
            for epoch in range(self.args.epochs):
                self.model_b.train()
                self.model_d.train()
                self.reset_meter(epoch)
                for _, data_tuple in tqdm(enumerate(self.train_loader)):
                    data = data_tuple[0].to(self.device)
                    label = data_tuple[1].to(self.device)
                    bias_label = data_tuple[2]
                    idx = data_tuple[3]

                    logit_b = self.model_b(data)
                    loss_b = self.bias_criterion(logit_b, label)
                    loss = loss_b.mean()

                    self.optimizer_b.zero_grad()
                    loss.backward()
                    self.optimizer_b.step()

                    acc = t.mean((logit_b.max(1)[1] == label).float().detach().cpu())

                    self.b_train_loss.update(loss,len(label))
                    self.b_train_acc.update(acc, len(label))
                self.validate(bias = True)
                self.test(bias = True)
                self.print_result(type='bias')
                self.step_b.step()
            self.save_models('end',bias=True)
        self.load_models('end',bias=True)

        # Calcuate probability
        self.model_b.eval()
        target = ['fc'] if 'ResNet' in self.args.model else ['fc']
        grad = grad_feat_ext(self.model_b,target, len(self.train_loader.dataset))
        # grad = grad_feat_ext(self.model_b,target)

        blabel = t.zeros(len(self.train_loader.dataset))
        idxorder = t.zeros(len(self.train_loader.dataset))
        start, end = 0,0

        for bidx, data_tuple in tqdm(enumerate(self.train_loader)):
            data = data_tuple[0].to(self.device).requires_grad_(True)
            label = data_tuple[1].to(self.device)
            bias_label = data_tuple[2]
            idx = data_tuple[3]

            logit_b = grad(data)
            loss_b = self.loss_b(logit_b, label)
            for sample_idx in range(len(label)):
                self.optimizer_b.zero_grad()
                loss_b[sample_idx].backward(retain_graph = True)

            end = start + len(label)
            blabel[start:end] = bias_label.detach().cpu()
            idxorder[start:end] = idx.detach().cpu()
            start = end
            
        for name in grad.hook_list.keys():
            if 'fc' in name:
                grad_mat = grad.hook_list[name].data
                break

        if self.args.norm_scale == -1:
            norm_scale = float('inf')
        elif self.args.norm_scale == -2:
            norm_scale = 2
        else:
            norm_scale =self.args.norm_scale

        # norm_scale = self.args.norm_scale if self.args.norm_scale != -1 else float('inf')
        score = t.norm(grad_mat, p=norm_scale, dim=1, keepdim=False)
        
        if self.args.norm_scale == -2:
            score = score * score


        # Magnitude
        order = t.argsort(idxorder)
        label = label[order]
        blabel = blabel[order]
        mag = t.clamp(score[order],min=1e-8)
        inv_mag = 1./mag
        norm_inv_mag = inv_mag / t.sum(inv_mag)
        mag_prob = 1./norm_inv_mag / t.sum(1./norm_inv_mag)


        self.train_loader.dataset.update_prob(mag_prob)
        self.train_loader.dataset.prob_sample_on()
        
        if self.args.save_stats:
            gradient_analysis(label, blabel, mag, self.args.print)
            prob_analysis(label,blabel,mag_prob ,self.args.print)
        
        del(grad_mat)

        self.model_d.load_state_dict(self.model_b.state_dict())

        if not self.args.pretrained_dmodel:
            # Re-initialize
            self.best_loss = np.inf
            self.best_acc = 0
            # Debiased train
            for epoch in range(self.args.epochs):
                self.model_b.train()
                self.model_d.train()
                self.reset_meter(epoch)
                for _, data_tuple in tqdm(enumerate(self.train_loader)):
                    data = data_tuple[0].to(self.device)
                    label = data_tuple[1].to(self.device)
                    bias_label = data_tuple[2]
                    idx = data_tuple[3]

                    logit_d = self.model_d(data)
                    loss_d = self.loss_d(logit_d, label)
                    loss = loss_d.mean()

                    self.optimizer_d.zero_grad()
                    loss.backward()
                    self.optimizer_d.step()

                    acc = t.mean((logit_d.max(1)[1] == label).float().detach().cpu())

                    self.d_train_loss.update(loss,len(label))
                    self.d_train_acc.update(acc, len(label))
                self.validate(debias=True)
                self.test(debias=True)
                self.print_result(type='debias')
                self.step_d.step()
            self.save_models('end',debias=True)
        self.load_models('best',debias=True)



        # Calcuate probability
        self.model_d.eval()
        target = ['fc'] if 'ResNet' in self.args.model else ['fc']
        grad = grad_feat_ext(self.model_d,target, len(self.train_loader.dataset))
        
        blabel = t.zeros(len(self.train_loader.dataset))
        idxorder = t.zeros(len(self.train_loader.dataset))
        start, end = 0,0

        for bidx, data_tuple in tqdm(enumerate(self.train_loader)):
            data = data_tuple[0].to(self.device).requires_grad_(True)
            label = data_tuple[1].to(self.device)
            bias_label = data_tuple[2]
            idx = data_tuple[3]

            logit_d = grad(data)
            loss_d = self.loss_d(logit_d, label)
            for sample_idx in range(len(label)):
                self.optimizer_d.zero_grad()
                loss_d[sample_idx].backward(retain_graph = True)

            end = start + len(label)
            blabel[start:end] = bias_label.detach().cpu()
            idxorder[start:end] = idx.detach().cpu()
            start = end
            
        for name in grad.hook_list.keys():
            if 'fc' in name:
                grad_mat = grad.hook_list[name].data
                break
        t.save(grad_mat[:end], self.args.log_dir+'gradient.pt')

    def train_dpr(self):
        # Initialize
        self.bias_criterion = GeneralizedCELoss(q=self.args.q)
        self.best_loss = np.inf
        self.best_val_acc = 0
        self.best_acc = 0

        # Bias Train
        if not self.args.pretrained_bmodel:
            for epoch in range(self.args.epochs):
                self.model_b.train()
                self.model_d.train()
                self.reset_meter(epoch)
                for _, data_tuple in enumerate(self.train_loader):
                    data = data_tuple[0].to(self.device)
                    label = data_tuple[1].to(self.device)
                    bias_label = data_tuple[2]
                    idx = data_tuple[-1]

                    logit_b = self.model_b(data)
                    loss_b = self.bias_criterion(logit_b, label)
                    loss = loss_b.mean()

                    self.optimizer_b.zero_grad()
                    loss.backward()
                    self.optimizer_b.step()

                    acc = t.mean((logit_b.max(1)[1] == label).float().detach().cpu())

                    self.b_train_loss.update(loss,len(label))
                    self.b_train_acc.update(acc, len(label))
                if self.args.dataset == 'celeba':
                    self.validate_wga(bias = True)
                    self.test_wga(bias = True)
                else:
                    self.validate(bias = True)
                    self.test(bias = True)
                self.print_result(type='bias')
                self.step_b.step()
            self.save_models('end',bias=True)
            print('biased model saved !')
            exit()

        if self.args.save_sampling_prob:
            # Calculate probability
            self.model_b.eval()
            grad_mat = None
            
            idxorder = t.zeros(len(self.train_loader.dataset))
            start, end = 0,0

            # g = (y, a)
            p_y_a = t.zeros((len(self.train_loader.dataset), self.args.num_class, self.args.num_class)) # dim = N x C x C (examples, y, a)
            sampling_probability = t.zeros(len(self.train_loader.dataset))
            labels = None
            biased_predictions = None
            

            for bidx, data_tuple in enumerate(self.train_loader):
                data = data_tuple[0].to(self.device)
                label = data_tuple[1].to(self.device)
                bias_label = data_tuple[2]
                idx = data_tuple[-1]
                
                end = start + len(label)
                if labels is None:
                    labels = t.zeros(len(self.train_loader.dataset)).to(label.dtype)
                    biased_predictions = t.zeros(len(self.train_loader.dataset)).to(label.dtype)

                logit_b = self.model_b(data)
                logit_b = logit_b / self.args.tau
                prob_b = F.softmax(logit_b, dim=1)
                predicted_a = t.argmax(prob_b, dim=1)
                
                dl_dw = (1 - prob_b[t.arange(prob_b.shape[0]), label])
                if grad_mat is None:
                    grad_mat = t.zeros(len(self.train_loader.dataset))
                    grad_mat[t.arange(start, end)] = dl_dw.detach().cpu()
                else:
                    grad_mat[t.arange(start, end)] = dl_dw.detach().cpu()
                

                # p_y_a[t.arange(start, end), label] = prob_b.detach().cpu()
                labels[t.arange(start, end)] = label.detach().cpu()
                biased_predictions[t.arange(start, end)] = predicted_a.detach().cpu()

                idxorder[start:end] = idx.detach().cpu()
                start = end
            
            # p_y_a = t.mean(p_y_a, dim=0)
            # p_y = t.sum(p_y_a, dim=1)
            
            sampling_probability = grad_mat
            # compute sampling probability
            sampling_probability = t.clamp(sampling_probability, min=1e-8)
                
            del(grad_mat)

            order = t.argsort(idxorder)

            t.save({'labels': labels[order], 
                    'sampling_probability': sampling_probability[order]}, 
                    # 'p_y_a': p_y_a, 
                    self.args.log_dir+f'model_b_end_sampling-prob_tau{self.args.tau}.pt')
            
            print(f'save sampling probability using tau {self.args.tau}')
            # print('p_y_a: ', p_y_a)
            exit()

        else:
            if not os.path.exists(self.args.log_dir+f'model_b_end_sampling-prob_tau{self.args.tau}.pt'):
                print(f'there is no saved file with tau {self.args.tau}. Please save sampling probability using "--save_sampling_prob" first.')
                exit()
            saved_data = t.load(self.args.log_dir+f'model_b_end_sampling-prob_tau{self.args.tau}.pt')
            print(f'load sampling probability with tau {self.args.tau}')

            mag = saved_data['sampling_probability']
            inv_mag = 1./mag
            norm_inv_mag = inv_mag / t.sum(inv_mag)
            mag_prob = 1./norm_inv_mag / t.sum(1./norm_inv_mag)


        self.train_loader.dataset.update_prob(mag_prob)
        self.train_loader.dataset.prob_sample_on()
        
        self.model_d.load_state_dict(self.model_b.state_dict())

        if not self.args.pretrained_dmodel:
            # Re-initialize
            self.best_loss = np.inf
            self.best_val_acc = 0
            self.best_acc = 0
            # Debiased train
            for epoch in range(self.args.epochs):
                self.model_b.train()
                self.model_d.train()
                self.reset_meter(epoch)
                for _, data_tuple in enumerate(self.train_loader):
                    data = data_tuple[0].to(self.device)
                    label = data_tuple[1].to(self.device)
                    bias_label = data_tuple[2]
                    idx = data_tuple[-1]

                    logit_d = self.model_d(data)
                    loss_d = self.loss_d(logit_d, label)
                    loss = loss_d.mean()

                    self.optimizer_d.zero_grad()
                    loss.backward()
                    self.optimizer_d.step()

                    acc = t.mean((logit_d.max(1)[1] == label).float().detach().cpu())

                    self.d_train_loss.update(loss,len(label))
                    self.d_train_acc.update(acc, len(label))
                self.validate(debias=True)
                self.test(debias=True)
                self.print_result(type='debias')
                self.step_d.step()
            self.save_models('end',debias=True)
        self.load_models('best',debias=True)
