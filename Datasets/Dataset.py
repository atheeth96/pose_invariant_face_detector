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

    def __init__(self,dataframe,abs_path='/home/vahadaneabhi01/datalab/training-assets/R_medical/atheeth/M2FPA/'\
                 , transform = None):
        #num_triplets = batch size 128
        
        self.df=pd.read_csv(dataframe)
        self.abs_path=abs_path
        self.len=len(self.df)
        self.transform=transform
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        
        label=self.df.iloc[idx,0]
        
        input_path=os.path.join(self.abs_path,self.df.iloc[idx,1])
        gt_path=os.path.join(self.abs_path,self.df.iloc[idx,2])
        
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
        
        mean=torch.tensor(self.mean,dtype=input_image.dtype).unsqueeze(1).unsqueeze(2)
        mean=mean.repeat((1,input_image.size()[1],input_image.size()[2]))
        
        std=torch.tensor(self.std,dtype=input_image.dtype).unsqueeze(1).unsqueeze(2)
        std=std.repeat((1,input_image.size()[1],input_image.size()[2]))
  
        input_image=(input_image-mean)/std
        gt_image=(gt_image-mean)/std
        
        return {'input_image': input_image,
        'gt_image': gt_image}
    
    
    
class ToLmdb(object):
   
    def __call__(self, sample):
        input_image, gt_image = sample['input_image'], sample['gt_image']
        
        input_image=np.expand_dims(input_image,axis=0)
        gt_image=np.expand_dims(gt_image,axis=0)
        
        image=np.concatenate((input_image,gt_image),axis=0)
        
        return image
                             
                             
    
class ReNormalize(object):
    def __init__(self,mean,std):
        self.mean=mean
        self.std=std

    def __call__(self, gen_image):

        mean=torch.tensor(self.mean,dtype=gen_image.dtype).unsqueeze(1).unsqueeze(2).unsqueeze(0)
        mean=mean.repeat((gen_image.size()[0],1,gen_image.size()[2],gen_image.size()[3]))
        
        std=torch.tensor(self.std,dtype=gen_image.dtype).unsqueeze(1).unsqueeze(2).unsqueeze(0)
        std=std.repeat((gen_image.size()[0],1,gen_image.size()[2],gen_image.size()[3]))

        gen_image=gen_image*std+mean
            
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
