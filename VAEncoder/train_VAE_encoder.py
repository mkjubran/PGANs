import argparse
import torch
import torch.optim as optim
#import nets_encoder as netE
import torchvision
from torchvision.utils import make_grid
from engine_encoder import train_encoder, validate_encoder
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.utils import save_image

import shutil
import os
import pdb
import nets32 as nets
import utilsG
import data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help=' ring | mnist | stackedmnist | cifar10 ')
parser.add_argument('--ckptG', type=str, default='', help='a given checkpoint file for generator')
parser.add_argument('--logsigma_file', type=str, default='', help='a given file for logsigma for the generator')
parser.add_argument('--ckptE', type=str, default='', help='a given checkpoint file for VA encoder')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lrE', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta', type=float, default=1, help='beta1 for KLD in ELBO')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector for encoder')
parser.add_argument('--ngf', type=int, default=64, help='model parameters for encoder')
parser.add_argument('--ndf', type=int, default=64, help='model parameters for encoder')
parser.add_argument('--nc', type=int, default = 1, help='number of channels for encoder')
parser.add_argument('--nzg', type=int, default=100, help='size of the latent z vector for generator')
parser.add_argument('--ngfg', type=int, default=64, help='model parameters for generator')
parser.add_argument('--ndfg', type=int, default=64, help='model parameters for generator')
parser.add_argument('--ncg', type=int, default = 1, help='number of channels for generator')
parser.add_argument('--Ntrain', type=int, default=60000, help='training set size for stackedmnist')
parser.add_argument('--Ntest', type=int, default=10000, help='test set size for stackedmnist')
parser.add_argument('--save_imgs_folder', type=str, default='../../outputs', help='where to save generated images')
args = parser.parse_args()

##-- preparing folders to save results
def VAE_folders(args):
 if not os.path.exists(args.save_imgs_folder):
    os.makedirs(args.save_imgs_folder)
 else:
    shutil.rmtree(args.save_imgs_folder)
    os.makedirs(args.save_imgs_folder)

 if not os.path.exists(args.ckptE):
    os.makedirs(args.ckptE)
 else:
    shutil.rmtree(args.ckptE)
    os.makedirs(args.ckptE)

##-- loading and spliting datasets
def load_datasets(data,args,device):
 dat = data.load_data(args.dataset, '../../data' , args.batchSize, device=device, imgsize=args.imageSize, Ntrain=args.Ntrain, Ntest=args.Ntest)
 trainset = dat['X_train']
 testset = dat['X_test']
 return trainset, testset

##-- loading PGAN generator model with sigma
def load_generator_wsigma(nets,device): #decoder for VAE
 netG = nets.Generator(args).to(device)
 if args.ckptG != '':
    netG.load_state_dict(torch.load(args.ckptG))
 else:
   print('A valid ckptG for a pretrained PGAN generator must be provided')
 return netG

if __name__ == "__main__":
 ##-- run on the available GPU otherwise CPUs
 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 ##-- preparing folders to save results
 VAE_folders(args)

 ##-- loading and spliting datasets
 trainset, testset = load_datasets(data,args,device)

 ##-- loading PGAN generator model with sigma
 netG = load_generator_wsigma(nets,device)

 ##-- setup the VAE Encoder and encoder training parameters
 #netE = nets.ConvVAE(args).to(device)
 netE = nets.ConvVAEType2(args).to(device)
 optimizer = optim.Adam(netE.parameters(), lr=args.lrE)

 ##-- write to tensor board
 writer = SummaryWriter(args.ckptE)

 # a list to save all the reconstructed images in PyTorch grid format
 grid_images = []

 train_loss = []
 valid_loss = []

 k=1.4
 logsigmaG = k*torch.ones(args.imageSize**2).to(device)
 for epoch in range(args.epochs):
    print(f"Epoch {epoch+1} of {args.epochs}")

    train_epoch_loss, elbo, KLDcf, reconloss = train_encoder(
        netE, args, trainset, device, optimizer, netG, logsigmaG
    )

    valid_epoch_loss, recon_images = validate_encoder(
        netE, args, testset, device, netG, logsigmaG
    )

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)

    # save the reconstructed images from the validation loop
    save_image(recon_images.cpu(), f"{args.save_imgs_folder}/output{epoch}.jpg")

    # convert the reconstructed images to PyTorch image grid format
    image_grid = make_grid(recon_images.detach().cpu())

    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {valid_epoch_loss:.4f}")
    writer.add_scalar("elbo/KLDcf", KLDcf, epoch)
    writer.add_scalar("elbo/reconloss", reconloss, epoch)
    writer.add_scalar("elbo/elbo", elbo, epoch)
    writer.add_histogram('distribution centers/enc1', netE.enc1.weight, epoch)
    #writer.add_histogram('distribution centers/enc2', netE.enc2.weight, epoch)

    # write images to tensorboard
    img_grid_TB = torchvision.utils.make_grid(recon_images.detach().cpu())
    if epoch % 2 == 0:
        writer.add_image('recon_images', img_grid_TB, epoch)

    torch.save(netE.state_dict(), os.path.join(args.ckptE,'netE_presgan_MNIST_epoch_%s.pth'%(epoch)))

 writer.flush()
 writer.close()


