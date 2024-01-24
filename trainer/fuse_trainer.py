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
from tensorboardX import SummaryWriter
from model.FlowInterp import Backwarp 
import torch.distributed as dist
import torchvision.utils as vutils
from taming.modules.losses.lpips import LPIPS#(Learned Perceptual Image Patch Similarity)

import sys

# from utils.dataset_fuse import Dataset
from utils.dataset import Dataset
from utils.tools import set_seed, set_device , Progbar, postprocess
from utils.loss import MS_SSIM, AdversarialLoss, Perceptual
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
        self.adversarial_loss = set_device(AdversarialLoss(type='hinge'))
        self.SSIM = set_device(MS_SSIM())
        self.l1_loss = nn.L1Loss()
        self.dis_writer = None
        self.gen_writer = None
        self.warp = Backwarp()
        self.summary = {}
        if self.args.global_rank == 0 or (not self.args.distributed):
            self.gen_writer = SummaryWriter(os.path.join(self.args.save_dir, 'gen'))
            self.dis_writer = SummaryWriter(os.path.join(self.args.save_dir, 'dis'))

        if self.args.load_flow:
            flownet = importlib.import_module('model.'+self.args.flow_model_name)
            self.model_flow = set_device(flownet.FlowInterpNet())
            flow_data = torch.load(self.args.Interp_pre_train, map_location = lambda storage, loc: set_device(storage)) 
            self.model_flow.load_state_dict(flow_data['net'])
            self.model_flow.eval()

        net = importlib.import_module('model.'+args.model_name)

        self.netG = set_device(net.FlowFuseNet())
        self.netD = set_device(net.Discriminator())
        
        self.optimG = torch.optim.Adam(self.netG.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))
        self.optimD = torch.optim.Adam(self.netD.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2))

        self.load()
        
        if self.args.distributed:
            self.netG = DDP(self.netG, device_ids=[self.args.global_rank], output_device=self.args.global_rank, 
                            broadcast_buffers=True, find_unused_parameters=True)
            self.netD = DDP(self.netD, device_ids=[self.args.global_rank], output_device=self.args.global_rank, 
                            broadcast_buffers=True, find_unused_parameters=True)
            
    # process input and calculate loss every training epoch
    def _train_epoch(self):
        
        progbar = Progbar(len(self.train_dataset), width=20, stateful_metrics=['epoch', 'iter'])
        mae = 0
        self.adjust_learning_rate()
        record_hole_loss = 0
        record_valid_loss = 0
        record_dis_loss = 0
        record_dis_fake_loss = 0
        record_gen_loss = 0
        record_gen_model_loss = 0
        for i, (img1,img2, img3, inter_img, mask,name) in enumerate(self.train_loader):

            self.iteration += 1 
            end = time.time()
            img1,img2,img3,inter_img,mask = set_device([img1,img2,img3,inter_img,mask])
            
            mask_img = img2*(1-mask)+mask
            
            with torch.no_grad():
                pyramid_flow = self.model_flow(inter_img, mask_img, mask)

            pred_result = self.netG(torch.cat([img1,self.warp(inter_img, pyramid_flow),img3],1),mask_img, mask)
            
            self.add_summary(self.gen_writer, 'lr/gen_lr', self.get_lr(type='G'))
            self.add_summary(self.dis_writer, 'lr/dis_lr', self.get_lr(type='D'))
            comp_img = pred_result * mask + img2 * (1-mask)
            gen_loss = 0
            dis_loss = 0
            # image discriminator loss
            dis_real_feat = self.netD(img2)    
            dis_fake_feat = self.netD(comp_img.detach())     
            dis_real_loss = self.adversarial_loss(dis_real_feat, True, True)
            dis_fake_loss = self.adversarial_loss(dis_fake_feat, False, True)
            dis_loss += (dis_real_loss + dis_fake_loss) / 2
            self.optimD.zero_grad()
            dis_loss.backward()
            self.optimD.step()

            # generator adversarial loss
            gen_fake_feat = self.netD(comp_img)                    # in: [rgb(3)]
            gen_fake_loss = self.adversarial_loss(gen_fake_feat, True, False) 
            gen_loss += gen_fake_loss * self.args.adv_weight
            recon_gen_loss = gen_fake_loss * self.args.adv_weight
            recon_hole_loss = self.l1_loss(pred_result*mask, img2*mask)/torch.mean(mask)
            recon_valid_loss = self.l1_loss(pred_result*(1-mask), img2*(1-mask))/torch.mean((1-mask))
            gen_loss += 6 * recon_hole_loss + 1* recon_valid_loss

            record_hole_loss += recon_hole_loss * 6
            record_valid_loss += recon_valid_loss * 1

            record_gen_model_loss += recon_gen_loss
            record_gen_loss += gen_loss
            record_dis_fake_loss += dis_fake_loss
            record_dis_loss += dis_loss
            
            # generator backward
            self.optimG.zero_grad()
            gen_loss.backward()
            self.optimG.step()      
            
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
                self.save_img(mask_img,inter_img, self.warp(inter_img, pyramid_flow), img1, img2, img3, pred_result, comp_img, name,inpaint_save_path)
        
        self.add_summary(self.gen_writer, 'loss/record_hole_loss', record_hole_loss.item()*self.args.batch_size/len(self.train_dataset))
        self.add_summary(self.gen_writer, 'loss/record_valid_loss', record_valid_loss.item()*self.args.batch_size/len(self.train_dataset))
        self.add_summary(self.gen_writer, 'loss/record_gen_loss', record_gen_loss.item()*self.args.batch_size/len(self.train_dataset))
        self.add_summary(self.gen_writer, 'loss/record_dis_loss', record_dis_loss.item()*self.args.batch_size/len(self.train_dataset))
        self.add_summary(self.gen_writer, 'loss/record_dis_fake_loss', record_dis_fake_loss.item()*self.args.batch_size/len(self.train_dataset))
        self.add_summary(self.gen_writer, 'loss/record_gen_model_loss', record_gen_model_loss.item()*self.args.batch_size/len(self.train_dataset))


    # get current learning rate
    def get_lr(self,type='G'):
        if type == 'G':
            return self.optimG.param_groups[0]['lr']
        return self.optimD.param_groups[0]['lr']

    # learning rate scheduler, step
    def adjust_learning_rate(self):
        decay = 0.1**(min(self.iteration, self.args.niter_steady) // self.args.niter) 
        new_lr = self.args.lr * decay
        if new_lr != self.get_lr():
            for param_group in self.optimG.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.optimD.param_groups:
                param_group['lr'] = new_lr
            
    def save_img(self, mask_image, image_inter, image_warp, image1, image2, image3, pred, comp_image, name, save_path_predict):
        num = mask_image.size(0)
        for i in range(num):
            if i % 15 == 0:
                inter_img = postprocess(image_inter)
                inter_img_path = os.path.join(save_path_predict,name[i].split('.')[0]+'inter_%03d.png'%(i))
                Image.fromarray(inter_img[i]).save(inter_img_path)

                warp_img = postprocess(image_warp)
                warp_img_path = os.path.join(save_path_predict,name[i].split('.')[0]+'warp_%03d.png'%(i))
                Image.fromarray(warp_img[i]).save(warp_img_path)

                mask_img = postprocess(mask_image)
                save_mask_img = mask_img[i]
                mask_img_path = os.path.join(save_path_predict,name[i].split('.')[0]+'mask_%03d.png'%(i))
                Image.fromarray(save_mask_img).save(mask_img_path)

                img1 = postprocess(image1)
                img1_path = os.path.join(save_path_predict,name[i].split('.')[0]+'img1_%03d.png'%(i))
                Image.fromarray(img1[i]).save(img1_path)

                img2 = postprocess(image2)
                img2_path = os.path.join(save_path_predict,name[i].split('.')[0]+'img2_%03d.png'%(i))
                Image.fromarray(img2[i]).save(img2_path)

                img3 = postprocess(image3)
                img3_path = os.path.join(save_path_predict,name[i].split('.')[0]+'img3_%03d.png'%(i))
                Image.fromarray(img3[i]).save(img3_path)

                pred_img = postprocess(pred)
                pred_img_path = os.path.join(save_path_predict,name[i].split('.')[0]+'pred_%03d.png'%(i))
                Image.fromarray(pred_img[i]).save(pred_img_path)

                comp_img = postprocess(comp_image)
                comp_img_path = os.path.join(save_path_predict,name[i].split('.')[0]+'comp_%03d.png'%(i))
                Image.fromarray(comp_img[i]).save(comp_img_path)
    

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
            gen_path = os.path.join(model_path, 'gen_{}.pth'.format(str(latest_epoch).zfill(5)))
            dis_path = os.path.join(model_path, 'dis_{}.pth'.format(str(latest_epoch).zfill(5)))
            opt_path = os.path.join(model_path, 'opt_{}.pth'.format(str(latest_epoch).zfill(5)))
            if self.args.global_rank == 0:
                print('Loading model from {}...'.format(gen_path))
            data = torch.load(gen_path, map_location = lambda storage, loc: set_device(storage)) 
            self.netG.load_state_dict(data['netG'])
            data = torch.load(dis_path, map_location = lambda storage, loc: set_device(storage)) 
            self.netD.load_state_dict(data['netD'])
            data = torch.load(opt_path, map_location = lambda storage, loc: set_device(storage))
            self.optimG.load_state_dict(data['optimG'])
            self.optimD.load_state_dict(data['optimD'])
            self.epoch = data['epoch']
            print('self.epoch is',self.epoch)
            self.iteration = data['iteration']
        else:
            if self.args.global_rank == 0:
                print('Warnning: There is no trained model found. An initialized model will be used.')

                
    # save parameters every eval_epoch
    def save(self, it):
        if self.args.global_rank == 0:
            gen_path = os.path.join(self.args.save_dir, 'gen_{}.pth'.format(str(it).zfill(5)))
            dis_path = os.path.join(self.args.save_dir, 'dis_{}.pth'.format(str(it).zfill(5)))
            opt_path = os.path.join(self.args.save_dir, 'opt_{}.pth'.format(str(it).zfill(5)))
            print('\nsaving model to {} ...'.format(gen_path))
            if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
                netG = self.netG.module
                netD = self.netD.module
            else:
                netG = self.netG
                netD = self.netD
            torch.save({'netG': netG.state_dict()}, gen_path)
            torch.save({'netD': netD.state_dict()}, dis_path)
            torch.save({'epoch': self.epoch, 
                        'iteration': self.iteration,
                        'optimG': self.optimG.state_dict(),
                        'optimD': self.optimD.state_dict()}, opt_path)
            os.system('echo {} > {}'.format(str(it).zfill(5), os.path.join(self.args.save_dir, 'latest.ckpt')))

    def add_summary(self, writer, name, val):
        if name not in self.summary:
            self.summary[name] = 0
        self.summary[name] += val
        if writer is not None and self.epoch % 1 == 0:
            writer.add_scalar(name, self.summary[name], self.epoch)
        self.summary[name] = 0
       
          
    def _valid_epoch(self, it):
        if self.config['global_rank'] == 0:
            print('[**] Testing in backend ...')
            model_path = self.config['save_dir']
            result_path = '{}/results_{}'.format(model_path, str(it).zfill(5))
            log = str(self.epoch) + 'valid.log'
            log_path = os.path.join(model_path, log)
            # log_path = os.path.join(model_path, str(self.epoch),'valid.log')

            os.popen('python test.py -c {} -n {} >> {};'
            'CUDA_VISIBLE_DEVICES=0,2 python eval.py -r {} >> {};'.format(self.config['config'], self.config['model_name'], log_path,
            result_path, log_path))
            file = open(log_path,'w')
            file.write(str(self.epoch))
            file.write('\n')

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
            
                if self.epoch > self.args.epoch:
                    break
            print('\nEnd training....')
        elif self.args.split == 'valid':
            self._test_epoch()
    
