import os

from PIL import Image

import torch
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image

import onnx
import onnxruntime



if __name__ == '__main__':
    
    img_path = 'demo_1.jpg' # Input image file
    out_path = 'pred_onnx_1.jpg' # Output image file
    model_path = '256x256_epoch_80.onnx'
   
    # Load and preprocess image
    img = Image.open(img_path)
    img = img.resize((256,256), resample=Image.LANCZOS)
    transform = transforms.Compose([
    transforms.Resize(256, Image.BICUBIC), transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])
    img = transform(img)
    img = img.unsqueeze(0)

    # Start an ONNX session
    ort_session = onnxruntime.InferenceSession(model_path)
    # Run the input through the model
    ort_inputs = {ort_session.get_inputs()[0].name: img.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)

    # Save output as an image
    save_image(torch.from_numpy(ort_outs[0]), out_path,  normalize=True)


