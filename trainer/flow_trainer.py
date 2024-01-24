import os
import time
import math
import glob
import shutil
import importlib
import datetime
import numpy as np
from PIL import Image
from math import log10

from functools import partial
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from model.FlowInterp import Backwarp 
from tensorboardX import SummaryWriter
import torch.distributed as dist
import torchvision.utils as vutils
from taming.modules.losses.lpips import LPIPS#(Learned Perceptual Image Patch Similarity)

import sys

from utils.dataset import Dataset
from utils.tools import set_seed, set_device , Progbar, postprocess
from utils.loss import MS_SSIM
import time

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class Trainer():

    def __init__(self, args):
        self.args = args
        self.epoch = 0
        self.iteration = 0
        
        # setup data set and data loader
        self.train_dataset = Dataset(args)
        worker_init_fn = partial(set_seed, base=args.seed)
        self.train_sampler = None
        if self.args.distributed:
            self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.args.world_size, rank=self.args.global_rank)
        self.train_loader = DataLoader(self.train_dataset, 
            batch_size= self.args.batch_size // self.args.world_size,
            shuffle=(self.train_sampler is None), num_workers=self.args.num_workers,
            pin_memory=True, sampler=self.train_sampler, worker_init_fn=worker_init_fn)
        self.batch_size= self.args.batch_size // self.args.world_size
        print('self.batch_size is',self.batch_size)
        # set up losses and metrics
        self.SSIM = set_device(MS_SSIM())
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = set_device(LPIPS().eval())
        self.flow_writer = None
        self.warp = Backwarp()
        self.step = args.step
        self.summary = {}
        if self.args.global_rank == 0 or (not self.args.distributed):
            self.flow_writer = SummaryWriter(os.path.join(self.args.save_dir, 'flow'))
        # self.train_args = self.config['trainer']

        net = importlib.import_module('model.'+args.model_name)
        self.net = set_device(net.FlowInterpNet())
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
        
        self.load()
        self.multiscale_weights = [1, 0.64, 0.32, 0.16]#, 0.08, 0.04]
        if self.args.distributed:
            print('distributed is true')
            self.net = DDP(self.net, device_ids=[self.args.global_rank], output_device=self.args.global_rank, 
                            broadcast_buffers=True, find_unused_parameters=True)

    # get current learning rate
    def get_lr(self):
        return self.optim.param_groups[0]['lr']
    
    # learning rate scheduler, step
    def adjust_learning_rate(self):
        baselr = self.args.lr
        total_epoch = self.epoch
        lr = baselr
        if total_epoch < 500:
            lr = baselr
        elif total_epoch < 800:
            lr = baselr / 2.
        elif total_epoch < 1200:
            lr = baselr / 4.
        elif total_epoch < 1400:
            lr = baselr / 8.
        
        
        new_lr = lr
        if new_lr != self.get_lr():
            print('new_lr is',new_lr)
            for param_group in self.optim.param_groups:
                param_group['lr'] = new_lr

    
    def tv_loss(self,x,epsilon=1e-6):
        loss = torch.mean( torch.sqrt(
            (x[:, :, :-1, :-1] - x[:, :, 1:, :-1]) ** 2 +
            (x[:, :, :-1, :-1] - x[:, :, :-1, 1:]) ** 2 + epsilon *epsilon
                )
            )
        return loss

    # load net
    def load(self):
        model_path = self.args.save_dir
        if os.path.isfile(os.path.join(model_path, 'latest.ckpt')):
            latest_epoch = open(os.path.join(model_path, 'latest.ckpt'), 'r').read().splitlines()[-1]
        else:
            ckpts = [os.path.basename(i).split('.pth')[0] for i in glob.glob(os.path.join(model_path, '*.pth'))]
            ckpts.sort()
            latest_epoch = ckpts[-1] if len(ckpts)>0 else None
        if latest_epoch is not None:
            flow_path = os.path.join(model_path, 'flow_{}.pth'.format(str(latest_epoch).zfill(5)))
            opt_path = os.path.join(model_path, 'opt_{}.pth'.format(str(latest_epoch).zfill(5)))
            if self.args.global_rank == 0:
                print('Loading model from {}...'.format(flow_path))
            
            data = torch.load(flow_path, map_location = lambda storage, loc: set_device(storage)) 
            self.net.load_state_dict(data['net'])
            data = torch.load(opt_path, map_location = lambda storage, loc: set_device(storage))
            self.optim.load_state_dict(data['optim'])
            self.epoch = data['epoch']
            print('self.epoch is',self.epoch)
            self.iteration = data['iteration']
        else:
            if self.args.global_rank == 0:
                print('Warnning: There is no trained model found. An initialized model will be used.')

                
    # save parameters every eval_epoch
    def save(self, it):
        if self.args.global_rank == 0:
            flow_path = os.path.join(self.args.save_dir, 'flow_{}.pth'.format(str(it).zfill(5)))
            opt_path = os.path.join(self.args.save_dir, 'opt_{}.pth'.format(str(it).zfill(5)))
            print('\nsaving model to {} ...'.format(flow_path))
            if isinstance(self.net, torch.nn.DataParallel) or isinstance(self.net, DDP):
                net = self.net.module
            else:
                net = self.net
            torch.save({'net': net.state_dict()}, flow_path)
            torch.save({'epoch': self.epoch, 
                        'iteration': self.iteration,
                        'optim': self.optim.state_dict()}, opt_path)
            os.system('echo {} > {}'.format(str(it).zfill(5), os.path.join(self.args.save_dir, 'latest.ckpt')))

    def add_summary(self, writer, name, val):
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if writer is not None and self.epoch % 1 == 0:
            writer.add_scalar(name, self.summary[name], self.epoch)
        self.summary[name] = 0
       

    # process input and calculate loss every training epoch
    def _train_epoch(self):
        
        progbar = Progbar(len(self.train_dataset), width=20, stateful_metrics=['epoch', 'iter'])
        mae = 0
        # self.adjust_learning_rate()
        
        record_hole_recon_loss = 0
        record_valid_recon_loss = 0
        record_SSIM_loss = 0
        record_perceptual_loss = 0
        record_gen_loss = 0
        for i, (img1, img2, img3, img_inter, mask,name) in enumerate(self.train_loader):
            self.iteration += 1
            # self.adjust_learning_rate()
            end = time.time()
            
            img1,img2,img3,img_inter,mask = set_device([img1,img2,img3,img_inter,mask])
            mask_img = img2*(1-mask)+mask
            
            pyramid_flow = self.net(img_inter, mask_img, mask)
            # predict flow from img1 to img2_mask
            self.add_summary(self.flow_writer, 'lr/flow_lr', self.get_lr())
            
            gen_loss = 0

            for f, weight in zip(pyramid_flow,self.multiscale_weights):
                f = F.interpolate(f, scale_factor=2, mode='bilinear', align_corners=True)
                resize_img_inter = F.interpolate(img_inter, size=f.size()[2:4], mode='bilinear', align_corners=True)
                resize_img2 = F.interpolate(img2, size=f.size()[2:4], mode='bilinear', align_corners=True)
                resize_mask = F.interpolate(mask, size=f.size()[2:4], mode='bilinear', align_corners=True)
                
                locate_mask = torch.ones_like(f)[:,:1,:,:]

                map_img= self.warp(resize_img_inter,f)
                map_mask= torch.where(self.warp(locate_mask,f)>0.7,1,0)
                
                hole_loss = self.l1_loss(resize_img2*resize_mask, map_img*resize_mask) / torch.mean(resize_mask)
                valid_loss = self.l1_loss(resize_img2*map_mask*(1-resize_mask), map_img*map_mask*(1-resize_mask)) / torch.mean(map_mask.float()*(1-resize_mask))
                SSIM_loss = (1 - self.SSIM((resize_img2*map_mask+1)/2*255, (map_img*map_mask+1)/2*255))/torch.mean(map_mask.float())
                perceptual_loss = torch.mean(torch.squeeze(self.perceptual_loss(resize_img2*map_mask.contiguous(), map_img*map_mask.contiguous())))/torch.mean(map_mask.float())

                resize_gen_loss = 5 * hole_loss + 1 * valid_loss + SSIM_loss + perceptual_loss
            
                record_hole_recon_loss += hole_loss * 5 * weight
                record_valid_recon_loss += valid_loss * 1 * weight
                record_SSIM_loss += SSIM_loss * 1 * weight
                record_perceptual_loss += perceptual_loss * 1 * weight
                record_gen_loss += resize_gen_loss * weight
                gen_loss += resize_gen_loss

            # generator backward
            self.optim.zero_grad()
            gen_loss.backward()
            self.optim.step()      
            
            # logs
            speed = img1.size(0)/(time.time() - end)*self.args.world_size
            logs = [("epoch", self.epoch),("iter", self.iteration),("lr", self.get_lr()),
                ('mae', mae), ('samples/s', speed)]
            if self.args.global_rank == 0:
                progbar.add(len(img1)*self.args.world_size, values=logs \
                if self.args.verbosity else [x for x in logs if not x[0].startswith('l_')])
            # saving and evaluating
            if self.iteration % self.args.save_freq == 0:
                self.save(int(self.iteration//self.args.save_freq))
            if self.iteration > self.args.iterations:
                break
            
            #save_image
            if self.epoch % int(self.args.img_save_freq) == 0 and self.args.global_rank == 0 and i == 0:
                inpaint_file_name = 'epoch_{}/'.format(str(self.epoch).zfill(5))
                if not os.path.exists(self.args.img_save_dir):
                    os.makedirs(self.args.img_save_dir)
                inpaint_save_path = os.path.join(self.args.img_save_dir,inpaint_file_name)
                if not os.path.exists(inpaint_save_path):
                    os.makedirs(inpaint_save_path)
                

        self.add_summary(self.flow_writer, 'loss/record_valid_recon_loss', record_valid_recon_loss.item()*self.args.batch_size/len(self.train_dataset))
        self.add_summary(self.flow_writer, 'loss/record_hole_recon_loss', record_hole_recon_loss.item()*self.args.batch_size/len(self.train_dataset))
        self.add_summary(self.flow_writer, 'loss/record_SSIM_loss', record_SSIM_loss.item()*self.args.batch_size/len(self.train_dataset))
        self.add_summary(self.flow_writer, 'loss/record_perceptual_loss', record_perceptual_loss.item()*self.args.batch_size/len(self.train_dataset))
        self.add_summary(self.flow_writer, 'loss/record_gen_loss', record_gen_loss.item()*self.args.batch_size/len(self.train_dataset))

    def train(self):
        print('start training')
        if self.args.split == 'train':
            while True:
                self.epoch += 1
                start_time = time.time()
                if self.args.distributed:
                    self.train_sampler.set_epoch(self.epoch)
                self._train_epoch()
                print('time is',time.time()-start_time)
                print('train epoch is', self.epoch)
                if self.epoch > self.args.epoch:
                    break
            print('\nEnd training....')
        elif self.args.split == 'valid':
            self._test_epoch()
