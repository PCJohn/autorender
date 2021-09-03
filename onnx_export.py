import os

from PIL import Image
import torch
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
import torch.onnx

from config import *
from facades import get_facades_loader
from models import Generator
from utils import make_dirs, make_gifs_test

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def inference():

    # Inference Path #
    make_dirs(config.inference_path)

    # Prepare Data Loader #
    test_loader = get_facades_loader(config.data_dir, 'test', config.test_batch_size)

    # Prepare Generator #

    # Test #
    print("Pix2Pix | Generating facades images started...")
    for i, (input, target) in enumerate(test_loader):

        # Prepare Data #
        target = target.to(device)

        # Generate Fake Image #

        # Save Images #
        result = torch.cat((target, input, generated), dim=0)
        save_image(result,
                   os.path.join(config.inference_path, 'Pix2Pix_Results_%03d.png' % (i+1)),
                   nrow=8,
                   normalize=True)

    make_gifs_test("Pix2Pix", config.inference_path)


if __name__ == '__main__':
    
    img_path = 'demo_1.jpg'
    out_path = 'pred_1.jpg'
    model_path = '/home/prithvi/code/illumination/Image-to-Image-Translation/results/weights/Pix2Pix_Generator_Epoch_80.pkl'

    # Setup and load generator
    G = Generator().to(device)
    G.load_state_dict(torch.load(model_path))
    G.eval()

    # Random input
    rand_img = torch.randn(1, 3, 256, 256, requires_grad=True)
    rand_img = rand_img.to(device)

    # Load and preprocess image
    #img = Image.open(img_path)
    #img = img.resize((256,256), resample=Image.LANCZOS)
    #transform = transforms.Compose([
    #    transforms.Resize(256, Image.BICUBIC), transforms.ToTensor(), transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))])
    #img = transform(img)

    
    #generated = G(img)
    
    #import pdb; pdb.set_trace();

    torch.onnx.export(G, 
                      rand_img,
                      '256x256_epoch_80.onnx',
                      export_params=True,
                      opset_version=11)

