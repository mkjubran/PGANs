
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import model_Likelihood as model
import torchvision
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

import shutil
import os
import pdb
import nets
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

parser.add_argument('--ckptL', type=str, default='', help='a given checkpoint file for Likelihood')
parser.add_argument('--save_Likelihood', type=str, default='../../outputs', help='where to save Likelihood results')
args = parser.parse_args()

##-- preparing folders to save results
def likelihood_folders(args):
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

##-- loading and spliting datasets
def load_datasets(data,args,device):
 dat = data.load_data(args.dataset, '../../input' , args.batchSize, device=device, imgsize=args.imageSize, Ntrain=args.Ntrain, Ntest=args.Ntest)
 trainset = dat['X_train']
 testset = dat['X_test']
 return trainset, testset

##-- loading PGAN generator model
def load_generator(nets,device):
 netG = nets.Generator(args).to(device)
 netG.apply(utilsG.weights_init)
 if args.ckptG != '':
    netG.load_state_dict(torch.load(args.ckptG))
 else:
   print('A valid ckptG for a pretrained PGAN generator must be provided')
 return netG

##-- loading VAE encoder model
def load_encoder(args):
 if args.ckptE != '':
        model.load_state_dict(torch.load(args.ckptE))
 else:
        print('A valid ckptE for a pretrained encoder must be provided')
 return model


def dist(args, mu, logvar, mean, scale, data):
 imageSize = args.imageSize

 ##-- compute MVN full batch
 mvn = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(scale).reshape(1, imageSize*imageSize, imageSize*imageSize))
 log_pxz_mvn = mvn.log_prob(data.view(-1,imageSize*imageSize))

 ##-- sample from standard normal distribution
 std = torch.exp(0.5*logvar)
 std_b = torch.eye(std.size(1)).to(device)
 std_c = std.unsqueeze(2).expand(*std.size(), std.size(1))
 std_3d = std_c * std_b
 mvnz = torch.distributions.MultivariateNormal(mu, scale_tril=std_3d)
 pz_normal = torch.exp(mvnz.log_prob(zr))
 return log_pxz_mvn, pz_normal


if __name__ == "__main__":
 ##-- run on the available GPU otherwise CPUs
 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 ##-- setup the VAE Encoder
 model = model.ConvVAE(args).to(device)
 optimizer = optim.Adam(model.parameters(), lr=args.lrE)

 ##-- preparing folders to save results
 likelihood_folders(args)

 ##-- loading and spliting datasets
 trainset, testset = load_datasets(data,args,device)

 train_loss = []
 valid_loss = []

 ##-- loading PGAN generator model
 netG = load_generator(nets,device)

 ##-- loading VAE encoder model
 model = load_encoder(args)

 ##-- write to tensor board
 writer = SummaryWriter(args.ckptL)

 ##-- setting scale and selecting a random test sample
 imageSize = args.imageSize
 scale = 0.01*torch.ones(imageSize**2)
 scale = scale.to(device)
 i = torch.randint(0, len(testset),(1,1)) ## selection of the index of test image
 data = testset[i].view([1,1,imageSize,imageSize])
 data = data.to(device)

 model.train()
 running_loss = 0.0
 counter = 0
 train_loss = []
 overlap_loss = 0;
 epoch = 0;
 while (epoch <= args.epochs) and (overlap_loss >= 0):
        epoch +=1
        counter += 1
        optimizer.zero_grad()
        x_hat, mu, logvar, z, zr = model(data, netG)
        mean = x_hat.view(-1,imageSize*imageSize)

        log_pxz_mvn, pz_normal = dist(args, mu, logvar, mean, scale, data)

        ##-- definning overlap loss abd backpropagation 
        overlap_loss = -1*(log_pxz_mvn + pz_normal)
        overlap_loss.backward()
        running_loss += overlap_loss.item()
        optimizer.step()
        train_loss = running_loss / counter

        ##-- print training loss
        if epoch % 5 ==0:
           print(f"Train Loss at epoch {epoch}: {train_loss:.4f}")

        ##-- printing only the positive overlap loss (to avoid printing extremely low numbers after training coverage to low positive value)
        if overlap_loss > 0:
           writer.add_scalar("Train Loss", overlap_loss, epoch)

        ##-- write to tensorboard
        if epoch % 10 == 0:
            img_grid_TB = torchvision.utils.make_grid(torch.cat((data, x_hat), 0).detach().cpu())
            writer.add_image('True and recon_image', img_grid_TB, epoch)

 writer.flush()
 writer.close()
