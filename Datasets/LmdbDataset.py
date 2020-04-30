from skimage.transform import resize

from torchvision import transforms
import torch
from torch.utils.data import Dataset

from io import BytesIO

import lmdb
import numpy as np



class LmdbDataset(Dataset):
    def __init__(self, path, transform):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = '{}'.format(str(index).zfill(6)).encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = np.load(buffer)
        
        input_image=img[0]
        gt_image=img[1]
        
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
        
        mean=torch.tensor(self.mean).unsqueeze(1).unsqueeze(2)
        mean=mean.repeat((1,input_image.size()[1],input_image.size()[2]))
        
        std=torch.tensor(self.std).unsqueeze(1).unsqueeze(2)
        std=std.repeat((1,input_image.size()[1],input_image.size()[2]))
  
        input_image=(input_image-mean)/std
        gt_image=(gt_image-mean)/std
        
        return {'input_image': input_image,
        'gt_image': gt_image}
    
    
class ReNormalize(object):
    def __init__(self,mean,std):
        self.mean=mean
        self.std=std

    def __call__(self, gen_image):

        mean=torch.tensor(self.mean).unsqueeze(1).unsqueeze(2).unsqueeze(0)
        mean=mean.repeat((gen_image.size()[0],1,gen_image.size()[2],gen_image.size()[3]))
        
        std=torch.tensor(self.std).unsqueeze(1).unsqueeze(2).unsqueeze(0)
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
