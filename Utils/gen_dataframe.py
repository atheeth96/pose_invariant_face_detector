import pandas as pd
import os
from tqdm import tqdm
import re

classes=os.listdir('../Train')
df_Train=pd.DataFrame(columns=['class','input','GT'])
for cat in tqdm(classes):
    expressions=os.listdir('../Train/{}/White/'.format(cat))
    for exp in expressions:
        images=os.listdir('../Train/{}/White/{}'.format(cat,exp))
        for image in images:
            index = re.search(r"\d", image).start(0)
            camera_id=image.split('.')[0][index:]
            illumination=image.split('.')[0][:index]
            gt_image=illumination+'4-7.jpg'
            df_Train=df_Train.append({'class':cat,'input': os.path.join('Train/{}/White/{}'.format(cat,exp),image)\
                             , 'GT': os.path.join('Train/{}/White/{}'.format(cat,exp),gt_image)}, ignore_index=True)
            
        
    
df_Train.to_csv('../M2FPA_TRAIN.csv',index=False)