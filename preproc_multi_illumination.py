import os
import cv2
import json
import numpy as np
from matplotlib import pyplot as plt

import multilum
from lanczos import resize_lanczos

INPUT_SIZE = 384
OUTPUT_SIZE = 256

RAW_DATA_DIR = '/home/prithvi/data/illumination/'
OUTPUT_DIR = '/home/prithvi/data/illumination/preproc_pix2pix_out256'


if __name__ == '__main__':
    
    raw_train_dir = os.path.join(RAW_DATA_DIR, 'multi_illumination_train_mip2_jpg')
    raw_test_dir = os.path.join(RAW_DATA_DIR, 'multi_illumination_test_mip2_jpg')

    if not os.path.exists(OUTPUT_DIR):
        os.system('mkdir '+OUTPUT_DIR)

    scenes = os.listdir(raw_train_dir)
    index = 0
    for scene in scenes:
        I = multilum.query_images([scene])
        P = multilum.query_probes([scene], material='chrome')
        
        with open(os.path.join(raw_train_dir,scene,'meta.json'),'r') as f:
            scene_metadata = json.load(f)
        f.close()

        for i,(img,ann) in enumerate(zip(I[0],P[0])):
            # Crop out chrome and gray ball from scene
            chrome_height = scene_metadata['chrome']['bounding_box']['y']
            chrome_height = int(chrome_height / multilum.imshape(0)[0] * img.shape[0])
            gray_height = scene_metadata['gray']['bounding_box']['y']
            gray_height = int(gray_height / multilum.imshape(0)[0] * img.shape[0])
            crop_height = min([chrome_height,gray_height])
            img = img[:crop_height]
            #import pdb; pdb.set_trace();
            
            img = resize_lanczos(img, INPUT_SIZE, INPUT_SIZE)
            ann = resize_lanczos(ann, OUTPUT_SIZE, OUTPUT_SIZE)

            print('Saving sample', index)
            cv2.imwrite(os.path.join(OUTPUT_DIR,'img_'+str(index).zfill(4)+'.jpg'),img)
            cv2.imwrite(os.path.join(OUTPUT_DIR,'ann_'+str(index).zfill(4)+'.jpg'),ann)
            index += 1
        


