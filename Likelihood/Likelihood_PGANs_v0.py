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
#from engine import train
#from engine_v1 import train_Likelihood, validate
from utils import save_reconstructed_images, image_to_vid, save_loss_plot
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

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

parser.add_argument('--ckptL', type=str, default='', help='a given checkpoint file for Likelihood')
parser.add_argument('--save_Likelihood', type=str, default='../../outputs', help='where to save Likelihood results')
args = parser.parse_args()

#ckptG = args.ckptG
#logsigma_file = args.logsigma_file
ckptE = args.ckptE

matplotlib.style.use('ggplot')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the model
model = model.ConvVAE(args).to(device)

# set the learning parameters
optimizer = optim.Adam(model.parameters(), lr=args.lrE)

# a list to save all the reconstructed images in PyTorch grid format
grid_images = []


## Checking paths and folders
if not os.path.exists(args.save_Likelihood):
    os.makedirs(args.save_Likelihood)
else:
    shutil.rmtree(args.save_Likelihood)
    os.makedirs(args.save_Likelihood)

if not os.path.exists(args.ckptL):
     os.makedirs(args.ckptL)
else:
     shutil.rmtree(args.ckptL)
     os.makedirs(args.ckptL)

##loading and spliting data
dataset = 'mnist'
dat = data.load_data(dataset, '../../input' , args.batchSize, device=device, imgsize=args.imageSize, Ntrain=args.Ntrain, Ntest=args.Ntest)
trainset = dat['X_train']
testset = dat['X_test']
trainloader=[]
testloader=[]

train_loss = []
valid_loss = []

writer = SummaryWriter(args.ckptL)

scale = 0.01
#scale = scale.to(device)
for i in range(1): #range(len(X_testing)):
    print(f"Likelihood of testing image {i} of (len(X_testing)")

    #model.apply(model.weights_init)
    if ckptE != '':
        model.load_state_dict(torch.load(ckptE))
    else:
        print('A valid ckptE must be provided')

    model.train()
    running_loss = 0.0
    counter = 0
    for epoch in tqdm(range(0, args.epochs)):
        data = testset[i].to(device)
        counter += 1
        optimizer.zero_grad()
        pdb.set_trace()

        reconstruction, mu, logvar, z, zr = model(data, netG)
        mean = x_hat.view(-1,args.imageSize*args.imageSize)
        pdb.set_trace()

        ### MVN full batch
        mvn = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(scale).reshape(1, imageSize*imageSize, imageSize*imageSize))
        #print([mvn.batch_shape, mvn.event_shape])
        log_pxz_mvn = mvn.log_prob(x.view(-1,imageSize*imageSize))
        #pdb.set_trace()

        std = torch.exp(0.5*logvar) # standard deviation
        std_b = torch.eye(std.size(1)).to(device)
        std_c = std.unsqueeze(2).expand(*std.size(), std.size(1))
        std_3d = std_c * std_b
        mvnz = torch.distributions.MultivariateNormal(mu, scale_tril=std_3d)

        pz_normal = torch.exp(mvnz.log_prob(zr))
        pz_log_pxz_mvn = torch.dot(log_pxz_mvn,pz_normal)
        reconloss = pz_log_pxz_mvn

        beta = args.beta
        elbo = beta*KLDcf - reconloss

        loss = elbo
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        train_loss = running_loss / counter

        pdb.set_trace()
        train_loss.append(train_epoch_loss)
        print(f"Train Loss: {train_epoch_loss:.4f}")

        #writer.add_scalar("elbo/KLDcf", KLDcf, epoch)
        #writer.add_scalar("elbo/reconloss", reconloss, epoch)
        #writer.add_scalar("elbo/elbo", elbo, epoch)
        #writer.add_histogram('distribution centers/enc1', model.enc1.weight, epoch)
        #writer.add_histogram('distribution centers/enc2', model.enc2.weight, epoch)

        #torch.save(model.state_dict(), os.path.join(ckptE,'netE_presgan_MNIST_epoch_%s.pth'%(epoch)))

writer.flush()
writer.close()

