import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/home/vahadaneabhi01/datalab/training-assets/R_medical/atheeth/M2FPA')

from Datasets.Dataset import Dataset,Scale,ToTensor,visualize_loader,Normalize,ToLmdb
from torch.utils.data import DataLoader
import argparse
from io import BytesIO


import numpy as np
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn


def convert(img, quality=100):

    buffer = BytesIO()
    np.save(buffer,img)
    val = buffer.getvalue()
    
    return val



def prepare(path, dataloader):
    write_frequency=10000
    db=lmdb.open(path, map_size=1024 ** 4, readahead=False)
    txn = db.begin(write=True)
    
    total = 0
    loop=tqdm(enumerate(dataloader))
    for i,img in loop:
        loop.set_description('Image {}/{}'.format(i + 1,len(dataloader) ))
        key = '{}'.format(str(i).zfill(3)).encode('utf-8')
        txn.put(key, convert(img.numpy()))

        total += 1
        if i+1 % write_frequency == 0:
            print("{}/{}" .foramt(i+1, len(data_loader)))
            txn.commit()
            txn = db.begin(write=True)


    txn.put('length'.encode('utf-8'), str(total).encode('utf-8'))
    txn.put('normalize'.encode('utf-8'), str('False').encode('utf-8'))


if __name__ == '__main__':
    
    transform=ToLmdb()
    batch_size=1
    train_dataset=Dataset('../M2FPA_TRAIN.csv',transform=transform)
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    
    prepare('Train.lmdb', train_loader)