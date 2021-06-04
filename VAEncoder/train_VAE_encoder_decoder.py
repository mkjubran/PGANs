import argparse
import torch
import torch.optim as optim
#import nets_encoder as netE
import torchvision
from torchvision.utils import make_grid
from engine_encoder import train_encoder_decoder, validate_encoder_decoder
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.utils import save_image
import torch.nn as nn
import datetime

import shutil
import os
import pdb
import nets
import utilsG
import data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help=' ring | mnist | stackedmnist | cifar10 ')
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
parser.add_argument('--ndfg', type=int, default=8, help='model parameters for generator')
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
 dat = data.load_data(args.dataset, '../../input' , args.batchSize, device=device, imgsize=args.imageSize, Ntrain=args.Ntrain, Ntest=args.Ntest)
 trainset = dat['X_train']
 testset = dat['X_test']
 return trainset, testset

if __name__ == "__main__":
 ##-- run on the available GPU otherwise CPUs
 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 ##-- preparing folders to save results
 VAE_folders(args)

 ##-- loading and spliting datasets
 trainset, testset = load_datasets(data,args,device)

 ##-- setup the VAE Encoder and encoder training parameters
 netE = nets.MixVAEncoderDecoder(args).to(device)
 optimizer = optim.Adam(netE.parameters(), lr=args.lrE)

 ##-- write to tensor board
 writer = SummaryWriter(args.ckptE+"/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
 criterion = nn.BCELoss(reduction='sum')

 # a list to save all the reconstructed images in PyTorch grid format
 grid_images = []

 train_loss = []
 valid_loss = []
 for epoch in range(args.epochs):
    print(f"Epoch {epoch+1} of {args.epochs}")

    train_epoch_loss, train_recon_images = train_encoder_decoder(
        netE, args, trainset, device, optimizer, criterion
    )

    valid_epoch_loss, recon_images = validate_encoder_decoder(
        netE, args, testset, device, criterion
    )

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)

    # save the reconstructed images from the validation loop
    save_image(recon_images.cpu(), f"{args.save_imgs_folder}/output{epoch}.jpg")

    # convert the reconstructed images to PyTorch image grid format
    image_grid = make_grid(recon_images.detach().cpu())

    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {valid_epoch_loss:.4f}")

    # write images to tensorboard
    img_grid_TB = torchvision.utils.make_grid(recon_images.detach().cpu())
    if epoch % 1 == 0:
        writer.add_image('recon_images', img_grid_TB, epoch)


    # write images to tensorboard
    img_grid_TB = torchvision.utils.make_grid(train_recon_images.detach().cpu())
    if epoch % 1 == 0:
        writer.add_image('train_recon_images', img_grid_TB, epoch)

    writer.add_scalar('train_loss', train_epoch_loss, epoch)
    writer.add_scalar('valid_loss', valid_epoch_loss, epoch)

    '''
    writer.add_scalar('distribution centers/enc1', netE.enc1.weight.mean().item(), epoch)
    writer.add_scalar('distribution centers/enc2', netE.enc2.weight.mean().item(), epoch)
    writer.add_scalar('distribution centers/dec1', netE.dec1.weight.mean().item(), epoch)
    writer.add_scalar('distribution centers/dec2', netE.dec2.weight.mean().item(), epoch)


    
    writer.add_histogram('distribution centers/enc1', netE.enc1.weight, epoch)
    writer.add_histogram('distribution centers/enc2', netE.enc2.weight, epoch)
    writer.add_histogram('distribution centers/enc3', netE.enc3.weight, epoch)
    writer.add_histogram('distribution centers/enc4', netE.enc4.weight, epoch)
    writer.add_histogram('distribution centers/dec1', netE.dec1.weight, epoch)
    writer.add_histogram('distribution centers/dec2', netE.dec2.weight, epoch)
    '''

    torch.save(netE.state_dict(), os.path.join(args.ckptE,'netEncDec_MNIST_epoch_%s.pth'%(epoch)))

 writer.flush()
 writer.close()


