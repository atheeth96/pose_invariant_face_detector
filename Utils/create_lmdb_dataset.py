import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/vahadaneabhi01/datalab/training-assets/R_medical/atheeth/M2FPA')
import os
import re
from Datasets.Dataset import Dataset,Scale,ToTensor,visualize_loader,Normalize,ToLmdb
from torch.utils.data import DataLoader
import argparse
from io import BytesIO
import multiprocessing
from skimage.transform import resize
from skimage.io import imread,imsave
import pandas as pd


import numpy as np
import lmdb
from tqdm import tqdm
import torchvision
from torchvision.transforms import functional as trans_fn


def convert(img, quality=100):

    buffer = BytesIO()
    np.save(buffer,img)
    val = buffer.getvalue()
    
    return val

def resize_and_convert(input_path, size=(256,256,3)):
    
    abs_path='/home/vahadaneabhi01/datalab/training-assets/R_medical/atheeth/M2FPA/'
    input_path=os.path.join(abs_path,input_path)
    
    input_image=imread(input_path)
    
    index = re.search(r"\d", input_path.split('/')[-1]).start(0)
    camera_id=input_path.split('/')[-1].split('.')[0][index:]
    illumination=input_path.split('/')[-1].split('.')[0][:index]
    gt_path='/'.join(input_path.split('/')[:-1])+'/'+illumination+'4-7.jpg'
    
    gt_image=imread(gt_path)
    
    input_image=(resize(input_image,size)*255).astype(np.uint8)
    gt_image=(resize(gt_image,size)*255).astype(np.uint8)
    
    image=np.concatenate((input_image,gt_image),axis=0)
    
    buffer = BytesIO()
    np.save(buffer, image)
    val = buffer.getvalue()

    return val


def prepare(path, path_list):
            
    with lmdb.open(path, map_size=1024 ** 4, readahead=False) as env:
        with env.begin(write=True) as txn:
            with multiprocessing.Pool(8) as pool:
                loop=tqdm(enumerate(pool.imap_unordered(resize_and_convert, path_list)))
                total = 0
                for i, img in loop:
                    loop.set_description('Image {}/{}'.format(i + 1,len(path_list) ))
                    
                    key = '{}'.format(str(i).zfill(6)).encode('utf-8')
                    txn.put(key, img)

                    total += 1

                txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))
                txn.put('normalize'.encode('utf-8'), str('False').encode('utf-8'))


if __name__ == '__main__':
    
    df=pd.read_csv('../M2FPA_TRAIN.csv')
    path_list=df['input'].tolist()
    
    prepare('../Train.lmdb', path_list)