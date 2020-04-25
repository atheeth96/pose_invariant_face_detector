from skimage.io import imread
import numpy as np
import os
import skimage
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import numpy as np
import pandas as pd
from skimage.transform import resize



    
class Dataset(Dataset):

    def __init__(self,dataframe, transform = None):
        #num_triplets = batch size 128
        
        self.df=pd.read_csv(dataframe)
        self.len=len(self.df)
        self.transform=transform
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        
        label=self.df.iloc[idx,0]
        
        input_path=self.df.iloc[idx,1]
        gt_path=self.df.iloc[idx,2]
        
        input_image=imread(input_path)
        gt_image=imread(gt_path)
        

        sample = {'input_image': input_image, 'gt_image':gt_image}

        if self.transform:
            sample = self.transform(sample)
        return sample
            

    
class Scale(object):
    """Convert ndarrays in sample to Tensors."""
   
    def __call__(self,sample):
        input_image, gt_image= sample['input_image'], sample['gt_image']
#         image_max=np.amax(image)

        input_image=(resize(input_image,(256,256,3))*255).astype(np.uint8)
        gt_image=(resize(gt_image,(256,256,3))*255).astype(np.uint8)
        
        input_image=input_image/255
        gt_image=gt_image/255
        
        
        return {'input_image': input_image, 'gt_image':gt_image}
    
    
class Normalize(object):
    def __init__(self,mean, std):
        self.mean=mean
        self.std=std

    def __call__(self, sample):
        input_image, gt_image = sample['input_image'], sample['gt_image']
        for i in range(input_image.shape[0]):

            input_image[i,:,:]=(input_image[i,:,:]-self.mean[i])/self.std[i]
            gt_image[i,:,:]=(gt_image[i,:,:]-self.mean[i])/self.std[i]
        return {'input_image': input_image,
        'gt_image': gt_image}
    
class ReNormalize(object):
    def __init__(self,mean,std):
        self.mean=mean
        self.std=std

    def __call__(self, gen_image):

        for i in range(gen_image.shape[0]):

            gen_image[i,:,:]=gen_image*self.std[i]+self.mean[i]
            
        gen_image=transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(gen_image)
            
        return gen_image 
    

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        input_image, gt_image= sample['input_image'], sample['gt_image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        
        input_image=input_image.transpose((2, 0, 1))
        
        gt_image=gt_image.transpose((2, 0, 1))
        
        
        
        return {'input_image': torch.from_numpy(input_image).type(torch.FloatTensor),
                'gt_image':torch.tensor(gt_image).type(torch.FloatTensor)}
    
    


    
def visualize_loader(loader,index=0):
    for i,sample in enumerate(loader):
        #print(sample['image'].shape)
        if i==index:
           
            input_image=(sample['input_image'][index]).numpy()
            gt_image=(sample['gt_image'][index]).numpy()
            
            print("MAX VALUE TENSOR: ","\ninput_image",np.max(input_image),"\ngt_image",np.max(gt_image))
            print("MIN VALUE TENSOR: ","\ninput_image",np.min(input_image),"\ngt_image",np.min(gt_image))
            print("TENSOR SIZE : ","\ninput_image",input_image.shape,"\ngt_image",gt_image.shape)
            
            
            input_image=input_image.transpose(1,2,0)
            input_image=((input_image*0.5+0.5)*255).astype(np.uint8)
            
            gt_image=gt_image.transpose(1,2,0)
            gt_image=((gt_image*0.5+0.5)*255).astype(np.uint8)
            
            return input_image,gt_image
