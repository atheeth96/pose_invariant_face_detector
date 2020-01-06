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



class NaiveFaceDataset(Dataset):

    def __init__(self, root_dir, csv_name, transform = None):
        #num_triplets = batch size 128
        
        self.root_dir=root_dir
        self.df=pd.read_csv(csv_name)
        self.label_set=set(self.df.Label.unique())
        self.transform=transform
        
    
    def __getitem__(self, idx):
        
        anchor_label=self.df.iloc[idx,1]
        anchor_path=os.path.join(self.root_dir,self.df.iloc[idx,0])
        
        pos_sample=self.df[self.df['Label']==anchor_label][self.df['ID']!=df.iloc[idx,0]]['ID'].sample().item()
        pos_path=os.path.join(self.root_dir,pos_sample)
        
        neg_label_list=list(self.label_set-set([pos_label]))
        neg_label=np.random.choice(neg_label_list)
        neg_sample=self.df[self.df['Label']==x]['ID'].sample().item()
        neg_path=os.path.join(self.root_dir,neg_sample)
        
     
        anchor_image=imread(anchor_path,as_gray=True)
        anchor_image=resize(anchor_image,(128,128))
        anchor_image=np.expand_dims(anchor_image,axis=2)
        
        pos_image=imread(pos_path,as_gray=True)
        pos_image=resize(pos_image,(128,128))
        pos_image=np.expand_dims(pos_image,axis=2)
        
        neg_image=imread(neg_path,as_gray=True)
        neg_image=resize(neg_image,(128,128))
        neg_image=np.expand_dims(neg_image,axis=2)
        
        
        sample = {'anc': anchor_image, 'pos':pos_image,'neg':neg_image}

        if self.transform:
            sample = self.transform(sample)
        return sample
            
    def __len__(self):
        
        return len(os.listdir(self.root_dir))
    
class NaiveScale(object):
    """Convert ndarrays in sample to Tensors."""
   
    def __call__(self,sample):
        anchor_image, pos_image,neg_image= sample['anc'], sample['pos'],sample['neg']
#         image_max=np.amax(image)
        anchor_image=anchor_image/255
        pos_image=pos_image/255
        neg_image=neg_image/255
        
        return {'anc': anchor_image, 'pos':pos_image,'neg':neg_image}

class NaiveToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        anchor_image, pos_image,neg_image= sample['anc'], sample['pos'],sample['neg']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        
        anchor_image=anchor_image.transpose((2, 0, 1))
        pos_image=pos_image.transpose((2, 0, 1))
        neg_image=neg_image.transpose((2, 0, 1))
        
        
        return {'anc': torch.from_numpy(anchor_image).type(torch.FloatTensor),
                'pos':torch.from_numpy(pos_image).type(torch.FloatTensor),
               'neg':torch.from_numpy(neg_image).type(torch.FloatTensor)}
    
    
    
class FaceDataset(Dataset):

    def __init__(self, root_dir, csv_name, transform = None):
        #num_triplets = batch size 128
        
        self.root_dir=root_dir
        self.df=pd.read_csv(csv_name)
        self.label_set=set(self.df.Label.unique())
        self.transform=transform
        
    
    def __getitem__(self, idx):
        
        label=self.df.iloc[idx,1]
        path=os.path.join(self.root_dir,self.df.iloc[idx,0])
        image=imread(path,as_gray=True)
        image=resize(image,(128,128))
        image=np.expand_dims(image,axis=2)
        
        
        
        
        sample = {'image': image, 'label':label}

        if self.transform:
            sample = self.transform(sample)
        return sample
            
    def __len__(self):
        
        return len(os.listdir(self.root_dir))
    
class Scale(object):
    """Convert ndarrays in sample to Tensors."""
   
    def __call__(self,sample):
        image, label= sample['image'], sample['label']
#         image_max=np.amax(image)
        image=image/255
        
        
        return {'image': image, 'label':label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label= sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        
        image=image.transpose((2, 0, 1))
        
        
        
        return {'image': torch.from_numpy(image).type(torch.FloatTensor),
                'label':torch.tensor(pos_image).type(torch.FloatTensor)}
    


    

