##Based on https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import model_v2 as model
import torchvision.transforms as transforms
import torchvision
import matplotlib
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from engine import train
from engine_v1 import train_PGAN, validate
from utils import save_reconstructed_images, image_to_vid, save_loss_plot
from torch.utils.tensorboard import SummaryWriter

import shutil
import os
import pdb
import nets
import utilsG
import data

parser = argparse.ArgumentParser()
parser.add_argument('--ckptG', type=str, default='', help='a given checkpoint file for generator')
parser.add_argument('--logsigma_file', type=str, default='', help='a given file for logsigma for the generator')
parser.add_argument('--ckptE', type=str, default='', help='a given checkpoint file for VA encoder')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
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

ckptG = args.ckptG
logsigma_file = args.logsigma_file
ckptE = args.ckptE

matplotlib.style.use('ggplot')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the model
model = model.ConvVAE(args).to(device)

# define and load the weights of the generator
logsigma_init = -1 #initial value for log_sigma_sian

#### defining generator
netG = nets.Generator(args).to(device)
#log_sigma = torch.tensor([logsigma_init]*(imageSize*imageSize), device=device, requires_grad=True)

#### initialize weights
netG.apply(utilsG.weights_init)
if ckptG != '':
    netG.load_state_dict(torch.load(ckptG))
if logsigma_file != '':
    logsigmaG = torch.load(logsigma_file)

# set the learning parameters
optimizer = optim.Adam(model.parameters(), lr=args.lrE)
criterion = nn.BCELoss(reduction='sum')

# a list to save all the reconstructed images in PyTorch grid format
grid_images = []

transform = transforms.Compose([
    transforms.Resize((args.imageSize, args.imageSize)),
    transforms.ToTensor(),
])


## Checking paths and folders
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

'''
# training set and train data loader
trainset = torchvision.datasets.MNIST(
    root='../../input', train=True, download=True, transform=transform
)
trainloader = DataLoader(
    trainset, batch_size=batch_size, shuffle=True
)
# validation set and validation data loader
testset = torchvision.datasets.MNIST(
    root='../../input', train=False, download=True, transform=transform
)
testloader = DataLoader(
    testset, batch_size=batch_size, shuffle=False
)

'''
##loading and spliting data
dataset = 'mnist'
dat = data.load_data(dataset, '../../input' , args.batchSize, device=device, imgsize=args.imageSize, Ntrain=args.Ntrain, Ntest=args.Ntest)
trainset = dat['X_train']
testset = dat['X_test']
trainloader=[]
testloader=[]

train_loss = []
valid_loss = []

writer = SummaryWriter(args.ckptE)

for epoch in range(args.epochs):
    print(f"Epoch {epoch+1} of {args.epochs}")
    train_epoch_loss, elbo, KLDcf, reconloss = train_PGAN(
        model, args, trainloader, trainset, device, optimizer, criterion, netG, logsigmaG
    )
    valid_epoch_loss, recon_images = validate(
        model, args, testloader, testset, device, criterion, netG, logsigmaG
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    # save the reconstructed images from the validation loop
    save_reconstructed_images(recon_images, epoch+1)
    # convert the reconstructed images to PyTorch image grid format
    image_grid = make_grid(recon_images.detach().cpu())
    #grid_images.append(image_grid)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {valid_epoch_loss:.4f}")

    writer.add_scalar("elbo/KLDcf", KLDcf, epoch)
    writer.add_scalar("elbo/reconloss", reconloss, epoch)
    writer.add_scalar("elbo/elbo", elbo, epoch)
    #pdb.set_trace()
    writer.add_histogram('distribution centers/enc1', model.enc1.weight, epoch)
    writer.add_histogram('distribution centers/enc2', model.enc2.weight, epoch)

    #writer.add_scalar("Train Loss", train_epoch_loss, epoch)
    #writer.add_scalar("Val Loss", valid_epoch_loss, epoch)

    # log images to tensorboard
    # create grid of images
    #pdb.set_trace()
    img_grid_TB = torchvision.utils.make_grid(recon_images.detach().cpu())

    # write to tensorboard
    if epoch % 2 == 0:
        writer.add_image('recon_images', img_grid_TB, epoch)

    torch.save(model.state_dict(), os.path.join(ckptE,'netE_presgan_MNIST_epoch_%s.pth'%(epoch)))

writer.flush()
writer.close()

# save the reconstructions as a .gif file
image_to_vid(grid_images)
# save the loss plots to disk
save_loss_plot(train_loss, valid_loss)
print('TRAINING COMPLETE')



