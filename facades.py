import os
import numpy as np
from glob import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from config import *

import albumentations as A
from albumentations.pytorch import ToTensorV2

class Facades(Dataset):
    """Facades Dataset"""
    def __init__(self, image_dir, purpose):

        self.image_dir = image_dir
        self.purpose = purpose
        self.imgs = [x for x in sorted(glob(os.path.join(self.image_dir, purpose, 'img_*.*')))]
        self.anns = [x for x in sorted(glob(os.path.join(self.image_dir, purpose, 'ann_*.*')))]
        
        """
        self.transform = transforms.Compose([
            transforms.Resize(config.crop_size, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        """
        #import pdb; pdb.set_trace();
        self.transform = A.Compose(
            [A.Resize(config.crop_size, config.crop_size, Image.BICUBIC),
             A.VerticalFlip(p=0.5),
             A.HorizontalFlip(p=0.5),
             A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, p=0.5),
             A.RGBShift(r_shift_limit=40, g_shift_limit=40, b_shift_limit=40, p=0.5),
             A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
             ToTensorV2()],
             additional_targets={'ann':'image'}
        )

    def __getitem__(self, index):
        img = Image.open(self.imgs[index])
        ann = Image.open(self.anns[index])
        width, height = img.size

        #img = self.transform(img)
        #ann = self.transform(ann)
        transformed = self.transform(image=np.asarray(img), ann=np.asarray(ann))
        img = transformed['image']
        ann = transformed['ann']

        return (img, ann)

    def __len__(self):
        return len(self.imgs)


def get_facades_loader(data_dir, purpose, batch_size):
    """Facades Data Loader"""
    if purpose == 'train':
        train_set = Facades(data_dir, 'train')
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        return train_loader

    elif purpose == 'val':
        val_set = Facades(data_dir, 'val')
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, drop_last=True)
        return val_loader

    elif purpose == 'test':
        test_set = Facades(data_dir, 'test')
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)
        return test_loader

    else:
        raise NameError('Purpose should be either train, val or test.')
