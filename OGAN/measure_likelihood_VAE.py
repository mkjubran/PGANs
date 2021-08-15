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
import utils
import utilsG
import data
import engine_OGAN
import copy
import statistics
import engine_PresGANs
import numpy as np
import datetime
import random
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--ckptDec', type=str, default='', help='a given checkpoint file for decoder')
parser.add_argument('--ckptE', type=str, default='', help='a given checkpoint file for VA encoder')
parser.add_argument('--seed_VAE', type=int, default=2019, help='data loading seed for VAE, default=2019')

parser.add_argument('--sample_from', type=str, default='generator', help='Sample from generator | dataset')
parser.add_argument('--save_likelihood_folder', type=str, default='../../outputs', help='where to save generated images')
parser.add_argument('--number_samples_likelihood', type=int, default=100, help='Number of Samples to considered to compute Likelihood')

parser.add_argument('--lrOL', type=float, default=0.001, help='learning rate for overlap loss, default=0.001')
parser.add_argument('--OLbatchSize', type=int, default=100, help='Overlap Loss batch size')
parser.add_argument('--S', type=int, default=1000, help='Sample Size when computing Likelihood')

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

##-- preparing likelihood folders to save results
def likelihood_folders(args):
 if not os.path.exists(args.save_likelihood_folder):
    os.makedirs(args.save_likelihood_folder)
 else:
    shutil.rmtree(args.save_likelihood_folder)
    os.makedirs(args.save_likelihood_folder)

##-- loading and spliting datasets
def load_datasets(data,args,device,seed):
 dat = data.load_data(args.dataset, '../../input' , args.batchSize, device=device, imgsize=args.imageSize, Ntrain=args.Ntrain, Ntest=args.Ntest, seed=seed)
 trainset = dat['X_train']
 testset = dat['X_test']
 return trainset, testset

##-- loading VAE decoder model
def load_decoder(netDec,ckptDec):
 if ckptDec != '':
        netDec.load_state_dict(torch.load(ckptDec))
 else:
        print('A valid ckptDec for a pretrained decoder must be provided')
 return netDec

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
def sample_from_generator(args,netG ,NoSamples):
 ##-- sample from standard normal distribution
 nz=100
 mean = torch.zeros(NoSamples,nz).to(device)
 scale = torch.ones(nz).to(device)
 mvn = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(scale).view(1, nz, nz))
 sample_z_shape = torch.Size([])
 sample_z = mvn.sample(sample_z_shape).view(-1,nz,1,1)
 recon_images = netG(sample_z)
 return recon_images

##-- get likelihood when sample from G1 and apply to E2,G2
def get_likelihood_sampleG1_applyE2G2(likelihood_G1_E2, args, device, netG1, netG2, logsigmaG2, netE2, netES, optimizerES, writer):

 #likelihood_G1_E2 = []
 samples_G1 = sample_from_generator(args, netG1, args.OLbatchSize) # sample from G1
 for i in range(args.OLbatchSize):
  netES.load_state_dict(copy.deepcopy(netE2.state_dict()))
  sample_G1 = samples_G1[i].view([1,1,args.imageSize,args.imageSize]).detach()
  likelihood_sample = engine_OGAN.get_likelihood(args,device,netES,optimizerES,sample_G1,netG2,logsigmaG2,args.save_likelihood_folder)
  likelihood_G1_E2.append(likelihood_sample.item())
  print(f"G1-->(E2,G2): sample {i} of {args.OLbatchSize}, OL = {likelihood_sample.item()}, moving mean = {statistics.mean(likelihood_G1_E2)}")

  #write moving average to TB
  writer.add_scalar("Moving Average/G1-->(E2,G2)", statistics.mean(likelihood_G1_E2), i)
 return likelihood_G1_E2

##-- get likelihood when sample from G2 and apply to E1,G1
def get_likelihood_sampleG2_applyE1G1(likelihood_G2_E1, args, device, netG2, netG1, logsigmaG1, netE1, netES, optimizerES, writer):

 #likelihood_G2_E1 = []
 samples_G2 = sample_from_generator(args, netG2, args.OLbatchSize) # sample from G2
 for i in range(args.OLbatchSize):
  netES.load_state_dict(copy.deepcopy(netE1.state_dict()))
  sample_G2 = samples_G2[i].view([1,1,args.imageSize,args.imageSize]).detach()
  likelihood_sample = engine_OGAN.get_likelihood(args,device,netES,optimizerES,sample_G2,netG1,logsigmaG1,args.save_likelihood_folder)
  likelihood_G2_E1.append(likelihood_sample.item())
  print(f"G2-->(E1,G1): sample {i} of {args.OLbatchSize}, OL = {likelihood_sample.item()}, moving mean = {statistics.mean(likelihood_G2_E1)}")

  #write moving average to TB
  writer.add_scalar("Moving Average/G2-->(E1,G1)", statistics.mean(likelihood_G2_E1), i)
 return likelihood_G2_E1

if __name__ == "__main__":
 ##-- run on the available GPU otherwise CPUs
 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 ##-- preparing folders to save results of Likelihood
 likelihood_folders(args)

 ##-- loading and spliting datasets
 trainset, testset = load_datasets(data,args,device, args.seed_VAE)

 ##-- loading VAE Decoder and setting decoder training parameters
 netDec = nets.VAEGenerator(args).to(device)
 #netDec = nets.LinearVADecoder(args).to(device)
 netDec = load_decoder(netDec,args.ckptDec)

 ##-- loading VAE Encoder and setting encoder training parameters
 netE = nets.ConvVAEType2(args).to(device)
 netE = load_encoder(netE,args.ckptE)
 optimizerE = optim.Adam(netE.parameters(), lr=args.lrE)

 ##-- setting scale and selecting a random test sample
 scale = 0.01*torch.ones(args.imageSize**2)
 scale = scale.to(device)
 #i = torch.randint(0, len(testset),(1,1)) ## selection of the index of test image

 ##-- define a new encoder netES to find OL per sample (need to keep the orogonal netE))
 netES = nets.ConvVAEType2(args).to(device)
 optimizerES = optim.Adam(netES.parameters(), lr=args.lrOL)

 testset= testset.to(device)
 trainset= trainset.to(device)

 log_dir = args.save_likelihood_folder+"/LResults_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
 writer = SummaryWriter(log_dir)

 Counter = 0
 likelihood_Dec = []
 samples_G1 = trainset[random.sample(range(0, len(trainset)), args.number_samples_likelihood)] 

 for j in range(0, args.number_samples_likelihood):
    Counter += 1

    ##-- compute OL where samples from dataset are applied to VAE Decoder
    sample_G1 = samples_G1[j].view([1,1,args.imageSize,args.imageSize]).detach()
    netES.load_state_dict(copy.deepcopy(netE.state_dict()))
    likelihood_sample = engine_OGAN.get_likelihood_VAE(args,device,netES,optimizerES,sample_G1,netDec,args.save_likelihood_folder)
    likelihood_Dec.append(likelihood_sample.item())
    print(f"Dataset-->VAE Dec: sample {Counter} of {args.number_samples_likelihood}, OL = {likelihood_sample.item()}, moving mean = {statistics.mean(likelihood_Dec)}")
    writer.add_scalar("Moving Average/Dataset-->VAE Dec", statistics.mean(likelihood_Dec), Counter)

 writer.flush()
 writer.close()
