import random
from random import shuffle
import os 
import numpy as np 
from PIL import Image
from glob import glob
import sys
import torch
import torchvision.transforms as transforms


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        super(Dataset, self).__init__()
        self.args = args
        self.split = self.args.split
        self.img_size = self.args.image_size
        
        self.data = []
        if self.split == 'train':
            self.img_data = [os.path.join(args.file_root, args.file_name, i) 
                for i in np.genfromtxt(os.path.join(args.flist_root, args.file_name, self.split+'.flist'), dtype=np.str_, encoding='utf-8')]
            for i in range(args.num_traindata//len(self.img_data)):
                self.data += self.img_data
            self.data += self.img_data[:self.args.num_traindata % len(self.img_data)]
            self.data.sort()
            shuffle(self.data)
        elif self.split == 'valid':
            self.img_data = [os.path.join(args.file_root, args.file_name, i) 
                for i in np.genfromtxt(os.path.join(args.flist_root, args.file_name, self.split+'.flist'), dtype=np.str_, encoding='utf-8')]
            self.data = self.img_data
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)
        return item

    # def set_subset(self, start, end):
    #     self.mask = self.mask[start:end]
    #     self.data = self.data[start:end] 

    def load_item(self, index):
        # load image
        img_path = os.path.dirname(self.data[index]) +'/'
        
        img1_name = 'img'+img_path[-4:-1]+'_1.png'
        img2_name = 'img'+img_path[-4:-1]+'_2.png'
        img3_name = 'img'+img_path[-4:-1]+'_3.png'
        img_inter_name = 'img'+img_path[-4:-1]+'inter_2.png'

       
        #image_read
        img1 = Image.open(img_path+img1_name).convert('RGB')
        img2 = Image.open(img_path+img2_name).convert('RGB')
        img3 = Image.open(img_path+img3_name).convert('RGB')
        img_inter = Image.open(img_path+img_inter_name).convert('RGB')

      
        mask = np.zeros([self.img_size,self.img_size]).astype(np.float32)
        mask_size = 0.3*self.img_size * self.img_size
        h_ = random.randint(int(0.4*self.img_size), int(0.6*self.img_size))
        # h_ = random.randint(int(0.6*self.img_size), int(0.8*self.img_size))
        w_ = int(mask_size//h_)
        mask[self.img_size//2-int(h_//2):self.img_size//2+int(h_//2),self.img_size//2-int(w_//2):self.img_size//2+int(w_//2)]=1
        mask = mask[np.newaxis,:,:]
        
        t1 = transforms.ToTensor()
        if self.split == 'train':
            self.Width, self.Height = img1.width, img1.height
            num1 = random.randint(0,self.Height - self.img_size)
            num2 = random.randint(0,self.Width - self.img_size)
            img1 = img1.crop((num2,num1,num2 + self.img_size, num1 + self.img_size))
            img1 = t1(img1)*2-1
            img2 = img2.crop((num2,num1,num2 + self.img_size, num1 + self.img_size))
            img2 = t1(img2)*2-1
            img3 = img3.crop((num2,num1,num2 + self.img_size, num1 + self.img_size))
            img3 = t1(img3)*2-1
            img_inter = img_inter.crop((num2,num1,num2 + self.img_size, num1 + self.img_size))
            img_inter = t1(img_inter)*2-1

        elif self.split == 'valid':
            img1 = t1(img1)*2-1
            img2 = t1(img2)*2-1
            img3 = t1(img3)*2-1
            img_inter = t1(img_inter)*2-1

        # if index % 3 == 0:
        #     return img3, img2, img1, mask, img1_name
        # else:
        #     return img1, img2, img3, mask, img1_name
        return img1, img2, img3, img_inter, mask, img1_name

   