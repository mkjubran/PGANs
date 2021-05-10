import argparse
import torch
import torch.optim as optim
import torchvision
from torchvision.utils import make_grid
from engine_encoder import train_encoder, validate_encoder
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.utils import save_image

import shutil
import os
import pdb
import nets
import utilsG
import data
import engine_OGAN
import copy
import statistics
import engine_PresGANs
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import seaborn as sns
import os
import pickle
import math
import utils
import hmc

from torch.distributions.normal import Normal

real_label = 1
fake_label = 0
criterion = nn.BCELoss()
criterion_mse = nn.MSELoss()

parser = argparse.ArgumentParser()
parser.add_argument('--ckptG1', type=str, default='', help='a given checkpoint file for generator 1')
parser.add_argument('--logsigma_file_G1', type=str, default='', help='a given file for logsigma for generator 1')
parser.add_argument('--lrG1', type=float, default=0.0002, help='learning rate for generator 1, default=0.0002')
parser.add_argument('--ckptD1', type=str, default='', help='a given checkpoint file for discriminator 1')
parser.add_argument('--lrD1', type=float, default=0.0002, help='learning rate for discriminator 1, default=0.0002')
parser.add_argument('--ckptE1', type=str, default='', help='a given checkpoint file for VA encoder 1')
parser.add_argument('--lrE1', type=float, default=0.0002, help='learning rate for encoder 1, default=0.0002')

parser.add_argument('--ckptG2', type=str, default='', help='a given checkpoint file for generator 2')
parser.add_argument('--logsigma_file_G2', type=str, default='', help='a given file for logsigma for generator 2')
parser.add_argument('--lrG2', type=float, default=0.0002, help='learning rate for generator 2, default=0.0002')
parser.add_argument('--ckptD2', type=str, default='', help='a given checkpoint file for discriminator 2')
parser.add_argument('--lrD2', type=float, default=0.0002, help='learning rate for discriminator 2, default=0.0002')
parser.add_argument('--ckptE2', type=str, default='', help='a given checkpoint file for VA encoder 2')
parser.add_argument('--lrE2', type=float, default=0.0002, help='learning rate for encoder 2, default=0.0002')

parser.add_argument('--ckptOL_E1', type=str, default='', help='a given checkpoint file for Overlap Loss - E1')
parser.add_argument('--save_OL_E1', type=str, default='../../outputs', help='where to save Overlap Loss results - E1')
parser.add_argument('--ckptOL_E2', type=str, default='', help='a given checkpoint file for Overlap Loss - E2')
parser.add_argument('--save_OL_E2', type=str, default='../../outputs', help='where to save Overlap Loss results - E2')
parser.add_argument('--ckptOL', type=str, default='', help='a given checkpoint file for Overlap Loss')

parser.add_argument('--ckptOL_G1I', type=str, default='', help='a given checkpoint file for G1 recon images with Overlap Loss G2 -->(E1,G1)')
parser.add_argument('--ckptOL_G2I', type=str, default='', help='a given checkpoint file for G2 recon imageswith Overlap Loss G1 --> (E2,G2)')
parser.add_argument('--ckptOL_G', type=str, default='', help='a given checkpoint file for G1 and G2 with Overlap Loss')

parser.add_argument('--lrOL', type=float, default=0.001, help='learning rate for overlap loss, default=0.001')
parser.add_argument('--OLbatchSize', type=int, default=100, help='Overlap Loss batch size')

parser.add_argument('--dataset', required=True, help=' ring | mnist | stackedmnist | cifar10 ')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--OLepochs', type=int, default=1000, help='number of epochs to train for Overlap Loss')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lrE', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta', type=float, default=1, help='beta for KLD in ELBO')
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

parser.add_argument('--W1', type=float, default=1, help='wight of OL of G2-->(E1,G1)')
parser.add_argument('--W2', type=float, default=1, help='wight of OL of G1-->(E2,G2)')

###### PresGAN-specific arguments
parser.add_argument('--sigma_lr', type=float, default=0.0002, help='generator variance')
parser.add_argument('--lambda_', type=float, default=0.01, help='entropy coefficient')
parser.add_argument('--sigma_min', type=float, default=0.01, help='min value for sigma')
parser.add_argument('--sigma_max', type=float, default=0.3, help='max value for sigma')
parser.add_argument('--logsigma_init', type=float, default=-1.0, help='initial value for log_sigma_sian')
parser.add_argument('--num_samples_posterior', type=int, default=2, help='number of samples from posterior')
parser.add_argument('--burn_in', type=int, default=2, help='hmc burn in')
parser.add_argument('--leapfrog_steps', type=int, default=5, help='number of leap frog steps for hmc')
parser.add_argument('--flag_adapt', type=int, default=1, help='0 or 1')
parser.add_argument('--delta', type=float, default=1.0, help='delta for hmc')
parser.add_argument('--hmc_learning_rate', type=float, default=0.02, help='lr for hmc')
parser.add_argument('--hmc_opt_accept', type=float, default=0.67, help='hmc optimal acceptance rate')
parser.add_argument('--save_sigma_every', type=int, default=1, help='interval to save sigma for sigan traceplot')
parser.add_argument('--stepsize_num', type=float, default=1.0, help='initial value for hmc stepsize')
parser.add_argument('--restrict_sigma', type=int, default=0, help='whether to restrict sigma or not')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--num_gen_images', type=int, default=10, help='number of images to generate for inspection')
parser.add_argument('--log', type=int, default=200, help='when to log')
parser.add_argument('--save_imgs_every', type=int, default=1, help='when to save generated images')

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

##-- preparing folders to save results of Likelihood
def OL_folders(args):
 if not os.path.exists(args.save_OL_E1):
    os.makedirs(args.save_OL_E1)
 else:
    shutil.rmtree(args.save_OL_E1)
    os.makedirs(args.save_OL_E1)

 if not os.path.exists(args.ckptOL_E1):
     os.makedirs(args.ckptOL_E1)
 else:
     shutil.rmtree(args.ckptOL_E1)
     os.makedirs(args.ckptOL_E1)

 if not os.path.exists(args.save_OL_E2):
    os.makedirs(args.save_OL_E2)
 else:
    shutil.rmtree(args.save_OL_E2)
    os.makedirs(args.save_OL_E2)

 if not os.path.exists(args.ckptOL_E2):
     os.makedirs(args.ckptOL_E2)
 else:
     shutil.rmtree(args.ckptOL_E2)
     os.makedirs(args.ckptOL_E2)

 if not os.path.exists(args.ckptOL):
     os.makedirs(args.ckptOL)
 else:
     shutil.rmtree(args.ckptOL)
     os.makedirs(args.ckptOL)

 if not os.path.exists(args.ckptOL_G):
     os.makedirs(args.ckptOL_G)
 else:
     shutil.rmtree(args.ckptOL_G)
     os.makedirs(args.ckptOL_G)

 if not os.path.exists(args.ckptOL_G1I):
     os.makedirs(args.ckptOL_G1I)
 else:
     shutil.rmtree(args.ckptOL_G1I)
     os.makedirs(args.ckptOL_G1I)

 if not os.path.exists(args.ckptOL_G2I):
     os.makedirs(args.ckptOL_G2I)
 else:
     shutil.rmtree(args.ckptOL_G2I)
     os.makedirs(args.ckptOL_G2I)


##-- loading and spliting datasets
def load_datasets(data,args,device):
 dat = data.load_data(args.dataset, '../../input' , args.batchSize, device=device, imgsize=args.imageSize, Ntrain=args.Ntrain, Ntest=args.Ntest)
 trainset = dat['X_train']
 testset = dat['X_test']
 return trainset, testset

##-- loading PGAN generator model with sigma
def load_generator_wsigma(netG,device,ckptG,logsigma_file):
 if ckptG != '':
    netG.load_state_dict(torch.load(ckptG))
 else:
   print('A valid ckptG for a pretrained PGAN generator must be provided')

 #logsigma_init = -1 #initial value for log_sigma_sian
 if logsigma_file != '':
    logsigmaG = torch.load(logsigma_file)
 else:
   print('A valid logsigma_file for a pretrained PGAN generator must be provided')
 return netG, logsigmaG

##-- loading PGAN discriminator model
def load_discriminator(netD,device,ckptD):
 if ckptD != '':
    netD.load_state_dict(torch.load(ckptD))
 else:
   print('A valid ckptD for a pretrained PGAN discriminator must be provided')
 return netD

##-- loading VAE encoder model
def load_encoder(netE,ckptE):
 if ckptE != '':
        netE.load_state_dict(torch.load(ckptE))
 else:
        print('A valid ckptE for a pretrained encoder must be provided')
 return netE

##-- loading get distribution
def dist(args, device, mu, logvar, mean, scale, data):
 imageSize = args.imageSize

 ##-- compute MVN full batch
 mvn = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(scale).reshape(1, imageSize*imageSize, imageSize*imageSize))
 log_pxz_mvn = mvn.log_prob(data.view(-1,imageSize*imageSize))

 ##-- computer sample from standard normal distribution
 std = torch.exp(0.5*logvar)
 std_b = torch.eye(std.size(1)).to(device)
 std_c = std.unsqueeze(2).expand(*std.size(), std.size(1))
 std_3d = std_c * std_b
 mvnz = torch.distributions.MultivariateNormal(mu, scale_tril=std_3d)
 pz_normal = torch.exp(mvnz.log_prob(zr))
 return log_pxz_mvn, pz_normal

##-- Sample from Generator
def sample_from_generator(args,netG):
 ##-- sample from standard normal distribution
 nz=100
 mean = torch.zeros(args.OLbatchSize,nz).to(device)
 scale = torch.ones(nz).to(device)
 mvn = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(scale).view(1, nz, nz))
 sample_z_shape = torch.Size([])
 sample_z = mvn.sample(sample_z_shape).view(-1,nz,1,1)
 recon_images = netG(sample_z)
 return recon_images

##-- get overlap loss when sample from G1 and apply to E2,G2
def OL_sampleG1_applyE2G2(args, device, netG1, netG2, netE2, netES, optimizerES, scale):
 #start = torch.cuda.Event(enable_timing=True)
 #end = torch.cuda.Event(enable_timing=True)
 #start.record()

 overlap_loss_G1_E2 = []
 samples_G1 = sample_from_generator(args, netG1) # sample from G1
 for i in range(args.OLbatchSize):
  # copy weights of netE2 to netES
  netES.load_state_dict(copy.deepcopy(netE2.state_dict()))
  sample_G1 = samples_G1[i].view([1,1,args.imageSize,args.imageSize]).detach()
  overlap_loss_sample = engine_OGAN.get_overlap_loss(args,device,netES,optimizerES,sample_G1,netG2,scale,args.ckptOL_E2)
  overlap_loss_G1_E2.append(overlap_loss_sample.item())
  print(f"G1-->(E2,G2): sample {i} of {args.OLbatchSize}, OL = {overlap_loss_sample.item()}, moving mean = {statistics.mean(overlap_loss_G1_E2)}")

  # write moving average to TB
  #writer.add_scalar("Moving Average/G1-->(E2,G2)", statistics.mean(overlap_loss_G1_E2), i)

 #end.record()
 #torch.cuda.synchronize()
 #print(start.elapsed_time(end))
 #print('Done G1 ---')
 return overlap_loss_G1_E2

##-- get overlap loss when sample from G2 and apply to E1,G1
def OL_sampleG2_applyE1G1(args, device, netG2, netG1, netE1, netES, optimizerES, scale):
 #start = torch.cuda.Event(enable_timing=True)
 #end = torch.cuda.Event(enable_timing=True)
 #start.record()

 overlap_loss_G2_E1 = []
 samples_G2 = sample_from_generator(args, netG2) # sample from G2
 for i in range(args.OLbatchSize):
  # copy weights of netE1 to netES
  netES.load_state_dict(copy.deepcopy(netE1.state_dict()))

  #sample_G2 = testset[i].view([1,1,imageSize,imageSize])
  sample_G2 = samples_G2[i].view([1,1,args.imageSize,args.imageSize]).detach()
  overlap_loss_sample = engine_OGAN.get_overlap_loss(args,device,netES,optimizerES,sample_G2,netG1,scale,args.ckptOL_E1)
  overlap_loss_G2_E1.append(overlap_loss_sample.item())
  #print(f"G2-->(E1,G1): sample {i} of {args.OLbatchSize}, OL = {overlap_loss_sample.item()}, moving mean = {statistics.mean(overlap_loss_G2_E1)}")

  # write moving average to TB
  #writer.add_scalar("Moving Average/G2-->(E1,G1)", statistics.mean(overlap_loss_G2_E1), i)

 #end.record()
 #torch.cuda.synchronize()
 #print(start.elapsed_time(end))
 #print('Done G2 ---')
 return overlap_loss_G2_E1

def distance_loss_G1_G2(netG1, netG2):

 G1_L0 = netG1.main[0].weight.view(100*512*4*4)
 G1_L3 = netG1.main[3].weight.view(512*256*4*4)
 G1_L6 = netG1.main[6].weight.view(256*128*4*4)
 G1_L9 = netG1.main[9].weight.view(128*64*4*4)
 G1_L12= netG1.main[12].weight.view(64*4*4)

 G2_L0 = netG2.main[0].weight.view(100*512*4*4)
 G2_L3 = netG2.main[3].weight.view(512*256*4*4)
 G2_L6 = netG2.main[6].weight.view(256*128*4*4)
 G2_L9 = netG2.main[9].weight.view(128*64*4*4)
 G2_L12= netG2.main[12].weight.view(64*4*4)

 #G1_L = torch.cat([G1_L6, G1_L9, G1_L12], dim=0)
 #G2_L = torch.cat([G2_L6, G2_L9, G2_L12], dim=0)

 G1_L = G1_L0
 G2_L = G2_L0

 G1_G2_L = (G1_L - G2_L)**2

 distance = G1_G2_L.sum().detach()
 #print(f"Ed(G1,G2)^2 = {distance}")
 return distance



if __name__ == "__main__":
 ##-- run on the available GPU otherwise CPUs
 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 ##-- preparing folders to save results of Likelihood
 OL_folders(args)

 ##-- loading and spliting datasets
 trainset, testset = load_datasets(data,args,device)

 ##-- loading PGAN generator model with sigma and setting generator training parameters - G1
 netG1 = nets.Generator(args).to(device)
 netG1, logsigmaG1 = load_generator_wsigma(netG1,device,args.ckptG1,args.logsigma_file_G1)
 optimizerG1 = optim.Adam(netG1.parameters(), lr=args.lrG1)
 sigma_optimizerG1 = optim.Adam([logsigmaG1], lr=args.sigma_lr, betas=(args.beta1, 0.999))

 ##-- loading PGAN generator model with sigma and setting generator training parameters - G2
 netG2 = nets.Generator(args).to(device)
 netG2, logsigmaG2 = load_generator_wsigma(netG2,device,args.ckptG2,args.logsigma_file_G2)
 optimizerG2 = optim.Adam(netG2.parameters(), lr=args.lrG2)
 sigma_optimizerG2 = optim.Adam([logsigmaG2], lr=args.sigma_lr, betas=(args.beta1, 0.999))

 ##-- loading PGAN discriminator model and setting discriminator training parameters - D1
 netD1 = nets.Discriminator(args).to(device)
 netD1 = load_discriminator(netD1,device,args.ckptD1)
 optimizerD1 = optim.Adam(netD1.parameters(), lr=args.lrD1)

 ##-- loading PGAN discriminator model and setting discriminator training parameters - D2
 netD2 = nets.Discriminator(args).to(device)
 netD2 = load_discriminator(netD2,device,args.ckptD2)
 optimizerD2 = optim.Adam(netD2.parameters(), lr=args.lrD2)

 ##-- loading VAE Encoder and setting encoder training parameters - E1
 netE1 = nets.ConvVAE(args).to(device)
 netE1 = load_encoder(netE1,args.ckptE1)
 optimizerE1 = optim.Adam(netE1.parameters(), lr=args.lrE1)

 ##-- loading VAE Encoder and setting encoder training parameters - E2
 netE2 = nets.ConvVAE(args).to(device)
 netE2 = load_encoder(netE2,args.ckptE2)
 optimizerE2 = optim.Adam(netE2.parameters(), lr=args.lrE2)

 ##-- setting scale and selecting a random test sample
 scale = 0.01*torch.ones(args.imageSize**2)
 scale = scale.to(device)
 #i = torch.randint(0, len(testset),(1,1)) ## selection of the index of test image

 # to estimate running time
 #start = torch.cuda.Event(enable_timing=True)
 #end = torch.cuda.Event(enable_timing=True)

 ##-- define a new encoder netES to find OL per sample (need to keep the orogonal netE))
 netES = nets.ConvVAE(args).to(device)
 optimizerES = optim.Adam(netES.parameters(), lr=args.lrOL)
 testset= testset.to(device)

 
 ##-- Write to tesnorboard
 writer = SummaryWriter(args.ckptOL_G)

 PresGANResultsG1=np.zeros(7)
 PresGANResultsG2=np.zeros(7)
 Counter_epoch_batch = 0
 for epoch in range(1, args.epochs+1):
  Counter = 0
  OLossG1 = 0
  OLossG2 = 0
  for j in range(0, len(trainset), args.batchSize):
    stop = min(args.batchSize, len(trainset[j:]))
    Counter += 1
    Counter_epoch_batch += 1

    #if ((Counter == 1) or (Counter % 10000000 == 0)):
    if Counter_epoch_batch % 100000000 == 0:
     
      ##-- compute OL where samples from G1 are applied to (E2,G2)
      overlap_loss_G1_E2 = OL_sampleG1_applyE2G2(args, device, netG1, netG2, netE2, netES, optimizerES, scale)
      OLossG2 = args.W2*(-1*statistics.mean(overlap_loss_G1_E2))
      OLossG2_No_W2 = (-1*statistics.mean(overlap_loss_G1_E2))

      ##-- compute OL where samples from G2 are applied to (E1,G1)
      overlap_loss_G2_E1 = OL_sampleG2_applyE1G1(args, device, netG2, netG1, netE1, netES, optimizerES, scale)
      OLossG1 = args.W1*(-1*statistics.mean(overlap_loss_G2_E1))
      OLossG1_No_W1 = (-1*statistics.mean(overlap_loss_G2_E1))

      TrueOLoss = OLossG1+OLossG2
      TrueOLoss_No_W1W2 = OLossG1_No_W1+OLossG2_No_W2

    ## compute the distance between G1 and G2 weights based on option#3
    Distance_G1G2 = (-1)*0.00000001*distance_loss_G1_G2(netG1, netG2) #option#3
    Distance_G1G2_No_W = distance_loss_G1_G2(netG1, netG2) #option#3

    ##-- OLoss is the use used to train the generators G1 and G2
    #OLoss = Distance_G1G2
    #OLoss = TrueOLoss
    OLoss = 0

    ##-- when to write to Tensorboard and harddrive the  images
    if Counter_epoch_batch % 20 == 0:
       save_imgs = True
    else:
       save_imgs = False

    ##-- update Generator 1 using Criterion = Dicriminator loss + W1*OverlapLoss(G2-->G1) + W2*OverlapLoss(G1-->G2)
    ##-----------------------
    writer = SummaryWriter(args.ckptOL_G1I)
    G1_X_training = trainset[j:j+stop].to(device)
    G1_fixed_noise = torch.randn(args.num_gen_images, args.nzg, 1, 1, device=device)

    if args.restrict_sigma:
        G1_logsigma_min = math.log(math.exp(args.sigma_min) - 1.0)
        G1_logsigma_max = math.log(math.exp(args.sigma_max) - 1.0)
    G1_stepsize = args.stepsize_num / args.nzg

    G1_bsz = args.batchSize
    G1_sigma_x = F.softplus(logsigmaG1).view(1, 1, args.imageSize, args.imageSize).to(device)

    netD1.zero_grad()
    G1_real_cpu = G1_X_training.to(device)

    G1_batch_size = G1_real_cpu.size(0)
    G1_label = torch.full((G1_batch_size,), real_label, device=device, dtype=torch.float32)

    G1_noise_eta = torch.randn_like(G1_real_cpu)
    G1_noised_data = G1_real_cpu + G1_sigma_x.detach() * G1_noise_eta
    G1_out_real = netD1(G1_noised_data)
    G1_errD_real = criterion(G1_out_real, G1_label)
    G1_errD_real.backward()
    G1_D_x = G1_out_real.mean().item()

    # train with fake
    G1_noise = torch.randn(G1_batch_size, args.nzg, 1, 1, device=device)
    G1_mu_fake = netG1(G1_noise) 
    G1_fake = G1_mu_fake + G1_sigma_x * G1_noise_eta
    G1_label.fill_(fake_label)
    G1_out_fake = netD1(G1_fake.detach())
    G1_errD_fake = criterion(G1_out_fake, G1_label)
    G1_errD_fake.backward()
    G1_D_G_z1 = G1_out_fake.mean().item()
    G1_errD = G1_errD_real + G1_errD_fake
    optimizerD1.step()

    # update G network: maximize log(D(G(z)))
    netG1.zero_grad()
    sigma_optimizerG1.zero_grad()
    G1_label.fill_(real_label)  
    G1_gen_input = torch.randn(G1_batch_size, args.nzg, 1, 1, device=device)
    G1_out = netG1(G1_gen_input)
    G1_noise_eta = torch.randn_like(G1_out)
    G1_g_fake_data = G1_out + G1_noise_eta * G1_sigma_x

    G1_dg_fake_decision = netD1(G1_g_fake_data)
    G1_g_error_gan = criterion(G1_dg_fake_decision, G1_label)+ OLoss 
    AdvLossG1 = G1_g_error_gan
    G1_D_G_z2 = G1_dg_fake_decision.mean().item()

    if args.lambda_ == 0:
       G1_g_error_gan.backward()
       optimizerG1.step() 
       sigma_optimizerG1.step()
    else:
       G1_hmc_samples, G1_acceptRate, G1_stepsize = hmc.get_samples(
                    netG1, G1_g_fake_data.detach(), G1_gen_input.clone(), G1_sigma_x.detach(), args.burn_in, 
                        args.num_samples_posterior, args.leapfrog_steps, G1_stepsize, args.flag_adapt, 
                            args.hmc_learning_rate, args.hmc_opt_accept)
                
       G1_bsz, G1_d = G1_hmc_samples.size()
       G1_mean_output = netG1(G1_hmc_samples.view(G1_bsz, G1_d, 1, 1).to(device))
       G1_bsz = G1_g_fake_data.size(0)

       G1_mean_output_summed = torch.zeros_like(G1_g_fake_data)
       for cnt in range(args.num_samples_posterior):
           G1_mean_output_summed = G1_mean_output_summed + G1_mean_output[cnt*G1_bsz:(cnt+1)*G1_bsz]
       G1_mean_output_summed = G1_mean_output_summed / args.num_samples_posterior  

       G1_c = ((G1_g_fake_data - G1_mean_output_summed) / G1_sigma_x**2).detach()
       G1_g_error_entropy = torch.mul(G1_c, G1_out + G1_sigma_x * G1_noise_eta).mean(0).sum()

       G1_g_error = G1_g_error_gan - args.lambda_ * G1_g_error_entropy
       G1_g_error.backward()
       optimizerG1.step() 
       sigma_optimizerG1.step()

    if args.restrict_sigma:
       G1_log_sigma.data.clamp_(min=G1_logsigma_min, max=G1_logsigma_max)

    print('G1: Epoch [%d/%d] .. Batch [%d/%d] .. Loss_D: %.4f .. Loss_G: %.4f .. D(x): %.4f .. D(G(z)): %.4f / %.4f'
                % (epoch, args.epochs, Counter, int(len(trainset)/args.batchSize), G1_errD.data, G1_g_error_gan.data, G1_D_x, G1_D_G_z1, G1_D_G_z2))

    DL = G1_errD.data
    GL = G1_g_error_gan.data
    Dx = G1_D_x
    DL_G_z1 = G1_D_G_z1
    DL_G_z2 = G1_D_G_z2

    PresGANResultsG1sample=[DL, GL, Dx, DL_G_z1, DL_G_z2, torch.min(G1_sigma_x), torch.max(G1_sigma_x)]

    if save_imgs:
        G1_fake = netG1(G1_fixed_noise).detach()
        vutils.save_image(G1_fake, '%s/G1_presgan_%s_fake_epoch_%03d.png' % (args.ckptOL_G1I, args.dataset, Counter_epoch_batch), normalize=True, nrow=20)
        img_grid = torchvision.utils.make_grid(G1_fake)
        writer.add_image('G1-fake_images', img_grid, Counter_epoch_batch)
    writer.flush()
    PresGANResultsG1 = PresGANResultsG1 + np.array(PresGANResultsG1sample)
    ##-----------------------


    ##-- update Generator 2 using Criterion = Dicriminator loss + W1*OverlapLoss(G2-->G1) + W2*OverlapLoss(G1-->G2)
    ##-----------------------

    writer = SummaryWriter(args.ckptOL_G2I)
    G2_X_training = trainset[j:j+stop].to(device)
    G2_fixed_noise = torch.randn(args.num_gen_images, args.nzg, 1, 1, device=device)

    if args.restrict_sigma:
        G2_logsigma_min = math.log(math.exp(args.sigma_min) - 1.0)
        G2_logsigma_max = math.log(math.exp(args.sigma_max) - 1.0)
    G2_stepsize = args.stepsize_num / args.nzg

    G2_bsz = args.batchSize
    G2_sigma_x = F.softplus(logsigmaG2).view(1, 1, args.imageSize, args.imageSize).to(device)

    netD2.zero_grad()
    G2_real_cpu = G2_X_training.to(device)

    G2_batch_size = G2_real_cpu.size(0)
    G2_label = torch.full((G2_batch_size,), real_label, device=device, dtype=torch.float32)

    G2_noise_eta = torch.randn_like(G2_real_cpu)
    G2_noised_data = G2_real_cpu + G2_sigma_x.detach() * G2_noise_eta
    G2_out_real = netD2(G2_noised_data)
    G2_errD_real = criterion(G2_out_real, G2_label)
    G2_errD_real.backward()
    G2_D_x = G2_out_real.mean().item()

    # train with fake
    G2_noise = torch.randn(G2_batch_size, args.nzg, 1, 1, device=device)
    G2_mu_fake = netG2(G2_noise) 
    G2_fake = G2_mu_fake + G2_sigma_x * G2_noise_eta
    G2_label.fill_(fake_label)
    G2_out_fake = netD2(G2_fake.detach())
    G2_errD_fake = criterion(G2_out_fake, G2_label)
    G2_errD_fake.backward()
    G2_D_G_z1 = G2_out_fake.mean().item()
    G2_errD = G2_errD_real + G2_errD_fake
    optimizerD2.step()

    # update G network: maximize log(D(G(z)))
    netG2.zero_grad()
    sigma_optimizerG2.zero_grad()
    G2_label.fill_(real_label)  
    G2_gen_input = torch.randn(G2_batch_size, args.nzg, 1, 1, device=device)
    G2_out = netG2(G2_gen_input)
    G2_noise_eta = torch.randn_like(G2_out)
    G2_g_fake_data = G2_out + G2_noise_eta * G2_sigma_x

    G2_dg_fake_decision = netD2(G2_g_fake_data)
    G2_g_error_gan = criterion(G2_dg_fake_decision, G2_label)+ OLoss 
    AdvLossG2 = G2_g_error_gan
    G2_D_G_z2 = G2_dg_fake_decision.mean().item()

    if args.lambda_ == 0:
       G2_g_error_gan.backward()
       optimizerG2.step() 
       sigma_optimizerG2.step()
    else:
       G2_hmc_samples, G2_acceptRate, G2_stepsize = hmc.get_samples(
                    netG2, G2_g_fake_data.detach(), G2_gen_input.clone(), G2_sigma_x.detach(), args.burn_in, 
                        args.num_samples_posterior, args.leapfrog_steps, G2_stepsize, args.flag_adapt, 
                            args.hmc_learning_rate, args.hmc_opt_accept)
                
       G2_bsz, G2_d = G2_hmc_samples.size()
       G2_mean_output = netG2(G2_hmc_samples.view(G2_bsz, G2_d, 1, 1).to(device))
       G2_bsz = G2_g_fake_data.size(0)

       G2_mean_output_summed = torch.zeros_like(G2_g_fake_data)
       for cnt in range(args.num_samples_posterior):
           G2_mean_output_summed = G2_mean_output_summed + mean_output[cnt*G2_bsz:(cnt+1)*G2_bsz]
       G2_mean_output_summed = G2_mean_output_summed / args.num_samples_posterior  

       G2_c = ((G2_g_fake_data - G2_mean_output_summed) / G2_sigma_x**2).detach()
       G2_g_error_entropy = torch.mul(G2_c, G2_out + G2_sigma_x * G2_noise_eta).mean(0).sum()

       G2_g_error = G2_g_error_gan - args.lambda_ * G2_g_error_entropy
       G2_g_error.backward()
       optimizerG2.step() 
       sigma_optimizerG2.step()

    if args.restrict_sigma:
       G2_log_sigma.data.clamp_(min=G2_logsigma_min, max=G2_logsigma_max)

    #print('G2: Epoch [%d/%d] .. Batch [%d/%d] .. Loss_D: %.4f .. Loss_G: %.4f .. D(x): %.4f .. D(G(z)): %.4f / %.4f'
    #            % (epoch, args.epochs, Counter, int(len(trainset)/args.batchSize), errD.data, g_error_gan.data, D_x, D_G_z1, D_G_z2))

    DL = G2_errD.data
    GL = G2_g_error_gan.data
    Dx = G2_D_x
    DL_G_z1 = G2_D_G_z1
    DL_G_z2 = G2_D_G_z2

    PresGANResultsG2sample=[DL, GL, Dx, DL_G_z1, DL_G_z2, torch.min(G2_sigma_x), torch.max(G2_sigma_x)]

    if save_imgs:
        G2_fake = netG2(G2_fixed_noise).detach()
        vutils.save_image(G2_fake, '%s/G2_presgan_%s_fake_epoch_%03d.png' % (args.ckptOL_G2I, args.dataset, Counter_epoch_batch), normalize=True, nrow=20)
        img_grid = torchvision.utils.make_grid(G2_fake)
        writer.add_image('G2-fake_images', img_grid, Counter_epoch_batch)
    writer.flush()
    PresGANResultsG2 = PresGANResultsG2 + np.array(PresGANResultsG2sample)

    ##-----------------------

    ##-- writing to Tensorboard
    if Counter_epoch_batch % 100000000 == 0:
       writer.add_scalar("Overlap Loss_batch/W1*OL[G2-->(E1,G1)]", OLossG1_No_W1, Counter_epoch_batch)
       writer.add_scalar("Overlap Loss_batch/W2*OL[G1-->(E2,G2)]", OLossG2_No_W2, Counter_epoch_batch)
       writer.add_scalar("Overlap Loss_batch/W2*OL[G2-->(E1,G1)] + W1*OL[G1-->(E2,G2)]", TrueOLoss_No_W1W2, Counter_epoch_batch)
       writer.add_scalar("Overlap Loss_batch/ Distance(G1,G2)", Distance_G1G2_No_W, Counter_epoch_batch)
       writer.add_scalar("Adversarial Loss/ AdvLoss G1", AdvLossG1, Counter_epoch_batch)
       #writer.add_scalar("Adversarial Loss/ AdvLoss G2", AdvLossG2, Counter_epoch_batch)
    

    if Counter_epoch_batch % 1 == 0:
       DL_G1 = PresGANResultsG1[0]/Counter_epoch_batch
       GL_G1 = PresGANResultsG1[1]/Counter_epoch_batch
       Dx_G1 = PresGANResultsG1[2]/Counter_epoch_batch
       DL_G1_z1 = PresGANResultsG1[3]/Counter_epoch_batch
       DL_G1_z2 = PresGANResultsG1[4]/Counter_epoch_batch
       sigma_x_G1_min = PresGANResultsG1[5]/Counter_epoch_batch
       sigma_x_G1_max = PresGANResultsG1[6]/Counter_epoch_batch

       DL_G2 = PresGANResultsG2[0]/Counter_epoch_batch
       GL_G2 = PresGANResultsG2[1]/Counter_epoch_batch
       Dx_G2 = PresGANResultsG2[2]/Counter_epoch_batch
       DL_G2_z1 = PresGANResultsG2[3]/Counter_epoch_batch
       DL_G2_z2 = PresGANResultsG2[4]/Counter_epoch_batch
       sigma_x_G2_min = PresGANResultsG2[5]/Counter_epoch_batch
       sigma_x_G2_max = PresGANResultsG2[6]/Counter_epoch_batch

       writer.add_scalar("G1-Loss/Loss_D", DL_G1, Counter_epoch_batch)
       writer.add_scalar("G1-Loss/Loss_G", GL_G1, Counter_epoch_batch)
       writer.add_scalar("G1-D(x)", Dx_G1, Counter_epoch_batch)
       writer.add_scalar("G1-DL_G/DL_G_z1", DL_G1_z1, Counter_epoch_batch)
       writer.add_scalar("G1-DL_G/DL_G_z2", DL_G1_z2, Counter_epoch_batch)
       writer.add_scalar("G1-sigma/sigma_min", sigma_x_G1_min, Counter_epoch_batch)
       writer.add_scalar("G1-sigma/sigma_max", sigma_x_G1_max, Counter_epoch_batch)

       writer.add_scalar("G2-Loss/Loss_D", DL_G2, Counter_epoch_batch)
       writer.add_scalar("G2-Loss/Loss_G", GL_G2, Counter_epoch_batch)
       writer.add_scalar("G2-D(x)", Dx_G2, Counter_epoch_batch)
       writer.add_scalar("G2-DL_G/DL_G_z1", DL_G2_z1, Counter_epoch_batch)
       writer.add_scalar("G2-DL_G/DL_G_z2", DL_G2_z2, Counter_epoch_batch)
       writer.add_scalar("G2-sigma/sigma_min", sigma_x_G2_min, Counter_epoch_batch)
       writer.add_scalar("G2-sigma/sigma_max", sigma_x_G2_max, Counter_epoch_batch)

       writer.flush()

