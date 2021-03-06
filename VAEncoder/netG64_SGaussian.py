import argparse
import torch
import nets
import pdb
import utilsG
import os
import shutil
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
#from utils import save_reconstructed_images_SG
from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('--ckptG', type=str, default='', help='a given checkpoint file for generator')
parser.add_argument('--nzg', type=int, default=100, help='size of the latent z vector for encoder')
parser.add_argument('--ngfg', type=int, default=64, help='model parameters for encoder')
parser.add_argument('--ndfg', type=int, default=64, help='model parameters for encoder')
parser.add_argument('--ncg', type=int, default = 1, help='number of channels for encoder')
args = parser.parse_args()
ckptG = args.ckptG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# define and load the weights of the generator
## define parameters
imageSize = 64 #the height / width of the input image to network'
nz =  100 # size of the latent z vector
ngf =  64
nc = 1 # number of channels on the data
logsigma_init = -1 #initial value for log_sigma_sian
iterations = 200
batch_size = 100

## load generator
print('Load Generator Weights')
netG = nets.Generator(args).to(device)

## initialize weights
netG.apply(utilsG.weights_init)
if ckptG != '':
    netG.load_state_dict(torch.load(ckptG))
else:
   print('Trained GAN Generator must be provided')

## Checking paths and folders to save images and TB staff
savefolder = '../../netG_SG'
if not os.path.exists(savefolder):
    os.makedirs(savefolder)
else:
    shutil.rmtree(savefolder)
    os.makedirs(savefolder)

## report to tensorboard
writer = SummaryWriter(savefolder)

## Input Standard Gaussian to netG
print('Create a Standard Multivariate Normal (MVN)')
mean = torch.zeros(batch_size,nz).to(device)
scale = torch.ones(nz).to(device)
mvn = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(scale).view(1, nz, nz))
sample_shape = torch.Size([])

print('Sampling from the Standard MVN and apply to Generator')
for iter in range(iterations):
   if iter % 10 == 0:
      print(f"Progress: {iter} / {iterations}")

   sample = mvn.sample(sample_shape).view(-1,nz,1,1)
   recon_images = netG(sample)

   ## write images to folder
   image_grid = make_grid(recon_images.detach().cpu())
   #save_reconstructed_images_SG(recon_images, iter, savefolder)
   save_image(recon_images.cpu(), '%s/output_SG_Sample%03d.png' % (savefolder, iter))

   ## write images to tensorboard
   img_grid_TB = torchvision.utils.make_grid(recon_images.detach().cpu())

   writer.add_image('recon_images', img_grid_TB, iter)

print(f"{iterations} images are generated and saved")

writer.flush()
writer.close()
