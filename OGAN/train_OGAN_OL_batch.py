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
import nets32 as nets
import utils
import utilsG
import data
import engine_OGAN_batch as engine_OGAN
import copy
import statistics
import engine_PresGANs
import numpy as np
import math
import random

#from torchvision.models.inception import inception_v3
import InceptionV3
import inception_score as iscore
import fid_score as fidscore

parser = argparse.ArgumentParser()
parser.add_argument('--ckptG1', type=str, default='', help='a given checkpoint file for generator 1')
parser.add_argument('--logsigma_file_G1', type=str, default='', help='a given file for logsigma for generator 1')
parser.add_argument('--lrG1', type=float, default=0.0002, help='learning rate for generator 1, default=0.0002')
parser.add_argument('--ckptD1', type=str, default='', help='a given checkpoint file for discriminator 1')
parser.add_argument('--lrD1', type=float, default=0.0002, help='learning rate for discriminator 1, default=0.0002')
parser.add_argument('--ckptE1', type=str, default='', help='a given checkpoint file for VA encoder 1')
parser.add_argument('--lrE1', type=float, default=0.0002, help='learning rate for encoder 1, default=0.0002')
parser.add_argument('--seed_G1', type=int, default=2019, help='data loading seed for generator 1, default=2019')

parser.add_argument('--ckptG2', type=str, default='', help='a given checkpoint file for generator 2')
parser.add_argument('--logsigma_file_G2', type=str, default='', help='a given file for logsigma for generator 2')
parser.add_argument('--lrG2', type=float, default=0.0002, help='learning rate for generator 2, default=0.0002')
parser.add_argument('--ckptD2', type=str, default='', help='a given checkpoint file for discriminator 2')
parser.add_argument('--lrD2', type=float, default=0.0002, help='learning rate for discriminator 2, default=0.0002')
parser.add_argument('--ckptE2', type=str, default='', help='a given checkpoint file for VA encoder 2')
parser.add_argument('--lrE2', type=float, default=0.0002, help='learning rate for encoder 2, default=0.0002')
parser.add_argument('--seed_G2', type=int, default=2020, help='data loading seed for generator 2, default=2020')

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
parser.add_argument('--S', type=int, default=1000, help='Sample Size when computing Likelihood')

parser.add_argument('--dataset', required=True, help=' ring | mnist | stackedmnist | cifar10 ')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--OLepochs', type=int, default=1000, help='number of epochs to train for Overlap Loss')
parser.add_argument('--overlap_loss_min', type=int, default=0, help='min value for Overlap Loss in VAE to determine the proposal')
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

parser.add_argument('--valbatches', type=int, default=100, help='Number of batches to use for validation')
parser.add_argument('--valevery', type=int, default=10000, help='Validate likelihood after training for valevery batches')
parser.add_argument('--mode', type=str, default='train_validate', help='Mode of operation [train, validate, train_validate]')

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

parser.add_argument('--GPU', type=int, default=0, help='GPU to use')

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
def load_datasets(data,args,device,seed):
 dat = data.load_data(args.dataset, '../../input' , args.batchSize, device=device, imgsize=args.imageSize, Ntrain=args.Ntrain, Ntest=args.Ntest, seed=seed)
 trainset = dat['X_train']
 testset = dat['X_test']
 return trainset, testset

##-- loading PGAN generator model with sigma
def load_generator_wsigma(netG,device,ckptG,logsigma_file):
 netG.apply(utils.weights_init)
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
 netD.apply(utils.weights_init)
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
def OL_sampleG1_applyE2G2(args, device, netG1, netG2, netE2, netES, optimizerE2, scale, logsigmaG2, netE2Orig):
 overlap_loss_G1_E2 = []
 samples_G1 = sample_from_generator(args, netG1) # sample from G1
 _, logvar_first, _, _ = netE2Orig(samples_G1, args)

 if True:
  likelihood_sample = engine_OGAN.get_likelihood_approx(args,device,netE2,optimizerE2,samples_G1,netG2,logsigmaG2,args.ckptOL_E2, logvar_first)
  overlap_loss_sample = likelihood_sample
  overlap_loss_G1_E2 = overlap_loss_sample
  print(f"G1-->(E2,G2) OL = {overlap_loss_sample}")
 return overlap_loss_G1_E2, netE2, optimizerE2

##-- get overlap loss when sample from G2 and apply to E1,G1
def OL_sampleG2_applyE1G1(args, device, netG2, netG1, netE1, netES, optimizerE1, scale, logsigmaG1, netE1Orig):
 overlap_loss_G2_E1 = []
 samples_G2 = sample_from_generator(args, netG2).detach() # sample from G2
 _, logvar_first, _, _ = netE1Orig(samples_G2, args)

 if True:
  likelihood_sample = engine_OGAN.get_likelihood_approx(args,device,netE1,optimizerE1,samples_G2,netG1,logsigmaG1,args.ckptOL_E1, logvar_first)
  overlap_loss_sample =  likelihood_sample
  overlap_loss_G2_E1 = overlap_loss_sample
  print(f"G2-->(E1,G1) OL = {overlap_loss_sample}")
 return overlap_loss_G2_E1, netE1, optimizerE1

def distance_loss_G1_G2(netG1, netG2):
 G1_L0 = netG1.main[0].weight.view(netG1.main[0].weight.shape[0]*netG1.main[0].weight.shape[1]*netG1.main[0].weight.shape[2]*netG1.main[0].weight.shape[3])

 G2_L0 = netG2.main[0].weight.view(netG2.main[0].weight.shape[0]*netG2.main[0].weight.shape[1]*netG2.main[0].weight.shape[2]*netG2.main[0].weight.shape[3])

 G1_L = G1_L0
 G2_L = G2_L0

 G1_G2_L = (G1_L - G2_L)**2

 distance = G1_G2_L.sum().detach()
 #print(f"Ed(G1,G2)^2 = {distance}")
 return distance



if __name__ == "__main__":
 ##-- run on the available GPU otherwise CPUs
 device = torch.device('cuda:'+str(args.GPU) if torch.cuda.is_available() else 'cpu')

 ##-- preparing folders to save results of Likelihood
 OL_folders(args)

 ##-- loading and spliting datasets for G1
 trainsetG1, testsetG1 = load_datasets(data,args,device, args.seed_G1)

 ##-- loading and spliting datasets for G2
 trainsetG2, testsetG2 = load_datasets(data,args,device, args.seed_G2)


 ##-- loading PGAN generator model with sigma and setting generator training parameters - G1
 netG1 = nets.Generator(args).to(device)
 netG1, logsigmaG1 = load_generator_wsigma(netG1,device,args.ckptG1,args.logsigma_file_G1)
 optimizerG1 = optim.Adam(netG1.parameters(), lr=args.lrG1, betas=(args.beta1, 0.999))
 sigma_optimizerG1 = optim.Adam([logsigmaG1], lr=args.sigma_lr, betas=(args.beta1, 0.999))

 ##-- loading PGAN generator model with sigma and setting generator training parameters - G2
 netG2 = nets.Generator(args).to(device)
 netG2, logsigmaG2 = load_generator_wsigma(netG2,device,args.ckptG2,args.logsigma_file_G2)
 optimizerG2 = optim.Adam(netG2.parameters(), lr=args.lrG2, betas=(args.beta1, 0.999))
 sigma_optimizerG2 = optim.Adam([logsigmaG2], lr=args.sigma_lr, betas=(args.beta1, 0.999))

 ##-- loading PGAN discriminator model and setting discriminator training parameters - D1
 netD1 = nets.Discriminator(args).to(device)
 netD1 = load_discriminator(netD1,device,args.ckptD1)
 optimizerD1 = optim.Adam(netD1.parameters(), lr=args.lrD1, betas=(args.beta1, 0.999))

 ##-- loading PGAN discriminator model and setting discriminator training parameters - D2
 netD2 = nets.Discriminator(args).to(device)
 netD2 = load_discriminator(netD2,device,args.ckptD2)
 optimizerD2 = optim.Adam(netD2.parameters(), lr=args.lrD2, betas=(args.beta1, 0.999))

 ##-- loading VAE Encoder and setting encoder training parameters - E1
 netE1 = nets.ConvVAEType2(args).to(device)
 netE1 = load_encoder(netE1,args.ckptE1)
 optimizerE1 = optim.Adam(netE1.parameters(), lr=args.lrE1)

 ##-- loading VAE Encoder and setting encoder training parameters - E2
 netE2 = nets.ConvVAEType2(args).to(device)
 netE2 = load_encoder(netE2,args.ckptE2)
 optimizerE2 = optim.Adam(netE2.parameters(), lr=args.lrE2)

 ##-- setting scale and selecting a random test sample
 scale = 0.01*torch.ones(args.imageSize**2)
 scale = scale.to(device)

 ##-- define a new encoder netES to find OL per sample (need to keep the orogonal netE))
 netES = nets.ConvVAEType2(args).to(device)
 optimizerES = optim.Adam(netES.parameters(), lr=args.lrOL)

 # create a copy of E1
 netE1Orig = nets.ConvVAEType2(args).to(device)
 netE1Orig.load_state_dict(copy.deepcopy(netE1.state_dict()))
 netE1Orig.eval()

 # create a copy of E2
 netE2Orig = nets.ConvVAEType2(args).to(device)
 netE2Orig.load_state_dict(copy.deepcopy(netE2.state_dict()))
 netE2Orig.eval()

 ##-- Load InceptionV3 model to compute FID
 block_idx = InceptionV3.InceptionV3.BLOCK_INDEX_BY_DIM[2048]
 inception_model = InceptionV3.InceptionV3([block_idx])
 inception_model=inception_model.to(device)

 ##-- Write to tesnorboard
 writer = SummaryWriter(args.ckptOL_G)

 PresGANResultsG1=np.zeros(10)
 PresGANResultsG2=np.zeros(10)
 Counter_epoch_batch = 0
 Counter_vald_epoch_batch = 0
 #Counter_G1test_E2 = 0
 #Counter_G2test_E1 = 0

 for epoch in range(1, args.epochs+1):
  Counter = 0
  OLossG1 = 0
  OLossG2 = 0

  for j in range(0, len(trainsetG1), args.batchSize):
    stop = min(args.batchSize, len(trainsetG1[j:]))
    Counter += 1
    Counter_epoch_batch += 1

    ##-- writing images to Tensorboard
    if (Counter_epoch_batch % 20 == 0) or (Counter_epoch_batch % args.valevery == 0) or (Counter_epoch_batch == 1):
       save_imgs = True
    else:
       save_imgs = False

    ##--------- Train G1
    if (args.mode == 'train') or (args.mode == 'train_validate'):
      ##-- measuer Ovelap loss to train G1
      if args.W1 != 0:
         ##-- compute OL where samples from G2 are applied to (E1,G1)
         overlap_loss_G2_E1, netE1, optimizerE1 = OL_sampleG2_applyE1G1(args, device, netG2, netG1, netE1, netES, optimizerE1, scale, logsigmaG1, netE1Orig)
         OLossG1 = args.W1*(overlap_loss_G2_E1)
         OLossG1_No_W1 = (overlap_loss_G2_E1)
      else:
         OLossG1 = 0
         overlap_loss_G2_E1 = 0
         OLossG1_No_W1 = 0

      if args.W2 != 0:
         ##-- compute OL where samples from G1 are applied to (E2,G2)
         overlap_loss_G1_E2, netE2, optimizerE2 = OL_sampleG1_applyE2G2(args, device, netG1, netG2, netE2, netES, optimizerE2, scale, logsigmaG2, netE2Orig)
         OLossG2 = args.W2*(overlap_loss_G1_E2)
         OLossG2_No_W2 = (overlap_loss_G1_E2)
      else:
         OLossG2 = 0
         overlap_loss_G1_E2=0
         OLossG2_No_W2 = 0

      ##-- compute the distance between G1 and G2 weights based on option#3
      Distance_G1G2 = (-1)*0.00000001*distance_loss_G1_G2(netG1, netG2) #option#3
      Distance_G1G2_No_W = distance_loss_G1_G2(netG1, netG2) #option#3

      ##-- OLoss is the use used to train the generators G1
      TrueOLoss = OLossG1+OLossG2
      TrueOLoss_No_W1W2 = OLossG1_No_W1+OLossG2_No_W2
      OLoss = TrueOLoss

      ##-- update Generator 1 using Criterion = Dicriminator loss + W1*OverlapLoss(G2-->G1) + W2*OverlapLoss(G1-->G2)
      netD1, netG1, logsigmaG1, AdvLossG1, PresGANResults, optimizerG1, optimizerD1, sigma_optimizerG1 = engine_PresGANs.presgan(args, device, epoch, trainsetG1[j:j+stop], netG1, optimizerG1, netD1, optimizerD1, logsigmaG1, sigma_optimizerG1, OLoss, args.ckptOL_G1I, save_imgs, 'G1', Counter_epoch_batch)
      PresGANResultsG1 = PresGANResultsG1 + np.array(PresGANResults)
      print('G1: Epoch [%d/%d] .. Batch [%d/%d] .. Loss_D: %.4f .. Loss_G: %.4f .. D(x): %.4f .. D(G(z)): %.4f / %.4f'
           % (epoch, args.epochs, Counter, int(len(trainsetG1)/args.batchSize), PresGANResults[0], PresGANResults[1], PresGANResults[2], PresGANResults[3], PresGANResults[4]))

      ##--------- Train G2
      ##-- measuer Ovelap loss to train G2
      if args.W1 != 0:
         ##-- compute OL where samples from G2 are applied to (E1,G1)
         overlap_loss_G2_E1, netE1, optimizerE1 = OL_sampleG2_applyE1G1(args, device, netG2, netG1, netE1, netES, optimizerE1, scale, logsigmaG1, netE1Orig)
         OLossG1 = args.W1*(overlap_loss_G2_E1)
         OLossG1_No_W1 = (overlap_loss_G2_E1)
      else:
         OLossG1 = torch.tensor(0)
         overlap_loss_G2_E1 = 0
         OLossG1_No_W1 = 0

      if args.W2 != 0:
         ##-- compute OL where samples from G1 are applied to (E2,G2)
         overlap_loss_G1_E2, netE2, optimizerE2 = OL_sampleG1_applyE2G2(args, device, netG1, netG2, netE2, netES, optimizerE2, scale, logsigmaG2, netE2Orig)
         OLossG2 = args.W2*(overlap_loss_G1_E2)
         OLossG2_No_W2 = (overlap_loss_G1_E2)
      else:
         OLossG2 = torch.tensor(0)
         overlap_loss_G1_E2=0
         OLossG2_No_W2 = 0

      ##-- compute the distance between G1 and G2 weights based on option#3
      Distance_G1G2 = (-1)*0.00000001*distance_loss_G1_G2(netG1, netG2) #option#3
      Distance_G1G2_No_W = distance_loss_G1_G2(netG1, netG2) #option#3

      ##-- OLoss is the use used to train the generators G2
      TrueOLoss = OLossG1+OLossG2
      TrueOLoss_No_W1W2 = OLossG1_No_W1+OLossG2_No_W2
      OLoss = TrueOLoss

      ##-- update Generator 2 using Criterion = Dicriminator loss + W1*OverlapLoss(G2-->G1) + W2*OverlapLoss(G1-->G2)
      netD2, netG2, logsigmaG2, AdvLossG2, PresGANResults, optimizerG2, optimizerD2, sigma_optimizerG2 = engine_PresGANs.presgan(args, device, epoch, trainsetG2[j:j+stop], netG2, optimizerG2, netD2, optimizerD2, logsigmaG2, sigma_optimizerG2, OLoss.detach(), args.ckptOL_G2I, save_imgs, 'G2', Counter_epoch_batch)
      PresGANResultsG2 = PresGANResultsG2 + np.array(PresGANResults)
      print('G2: Epoch [%d/%d] .. Batch [%d/%d] .. Loss_D: %.4f .. Loss_G: %.4f .. D(x): %.4f .. D(G(z)): %.4f / %.4f'
           % (epoch, args.epochs, Counter, int(len(trainsetG2)/args.batchSize), PresGANResults[0], PresGANResults[1], PresGANResults[2], PresGANResults[3], PresGANResults[4]))

    if (args.mode == 'train_validate') or (args.mode == 'validate'):
      ##-- validation step
      if (Counter_epoch_batch % args.valevery == 0) or (Counter_epoch_batch == 1) or (args.mode == 'validate'):
         Counter_G1test_E2 = 0
         Counter_G2test_E1 = 0
         ISsumG1=0; ISsumG2=0
         FIDsumG1=0; FIDsumG2=0
         Counter_vald_epoch_batch += 1
         
         MinNTest = min(testsetG1.shape[0],args.valbatches*args.OLbatchSize)
         samples_G1test = testsetG1[random.sample(range(0, len(testsetG1)), MinNTest)] 
         samples_G2test = testsetG2[random.sample(range(0, len(testsetG2)), MinNTest)]
         if args.valbatches*args.OLbatchSize > testsetG1.shape[0]:
            args.valbatches = int(testsetG1.shape[0]/args.OLbatchSize)


         for cnt in range(0, args.valbatches*args.OLbatchSize, args.OLbatchSize):
             #netE1 = nets.ConvVAEType2(args).to(device)
             #netE1 = load_encoder(netE1,args.ckptE1)
             #optimizerE1 = optim.Adam(netE1.parameters(), lr=args.lrE1)

             #netE2 = nets.ConvVAEType2(args).to(device)
             #netE2 = load_encoder(netE2,args.ckptE2)
             #optimizerE2 = optim.Adam(netE2.parameters(), lr=args.lrE2)

             ## Validation by measuring Likelihood of G2
             Counter_G1test_E2 += 1
             sample_G1 = samples_G1test[cnt:cnt+args.OLbatchSize].view([-1,args.nc,args.imageSize,args.imageSize]).detach().to(device)
             _, logvar_first, z, _ = netE2Orig(sample_G1, args)
             likelihood_sample = engine_OGAN.get_likelihood(args,device,netE2,optimizerE2,sample_G1,netG2,logsigmaG2,args.ckptOL_E2,logvar_first)
             if (cnt == 0) and (Counter_G1test_E2 == 1):
                likelihood_G1test_E2 = likelihood_sample.detach().view(1,1)
                likelihood_G1test_E2_mean = likelihood_G1test_E2
             else:
                likelihood_G1test_E2 = torch.cat((likelihood_G1test_E2,likelihood_sample.detach().view(1,1)),1)
                #likelihood_G1test_E2_mean = torch.logsumexp(likelihood_G1test_E2,1)-torch.log(torch.tensor(likelihood_G1test_E2.shape[1]))
                likelihood_G1test_E2_mean = torch.mean(likelihood_G1test_E2,1)
             LL_G1test_E2_mean=likelihood_G1test_E2_mean
             #print(f"Validation: G1(testset)-->(E2,G2)({Counter_epoch_batch}): batch {Counter_G1test_E2} of {int(args.valbatches)}, LL (batch) = {likelihood_sample.item()}, LL (moving average) = {LL_G1test_E2_mean.item()}")
             writer.add_scalar("Validation LL Sample/G1(testset)-->(E2,G2)", likelihood_sample.item(),  Counter_vald_epoch_batch*args.valbatches*args.OLbatchSize+cnt )

             ## Measuring Inception Score
             x_hat = netG2(z)
             if x_hat.shape[1] != 3:
                x_hat = x_hat.repeat([1,3,1,1])
             IS = iscore.inception_score(x_hat, device, batch_size=32, resize=True, splits=1)
             ISsumG2=(ISsumG2+IS[0]);ISmean_G2=ISsumG2/Counter_G1test_E2;

             ## Measuring FID score for sample_G1 (real) and x_hat=netG2(z) (fake)
             FID_G2 = fidscore.calculate_fretchet(sample_G1,x_hat,inception_model,device)
             FIDsumG2 = FIDsumG2 + FID_G2; FIDmean_G2=FIDsumG2/Counter_G1test_E2

             print(f"Validation: G1(testset)-->(E2,G2)({Counter_epoch_batch}): batch {Counter_G1test_E2} of {int(args.valbatches)}, LL (batch) = {likelihood_sample.item()}, LL (moving average) = {LL_G1test_E2_mean.item()}, IS = {IS[0]}, ISmean = {ISmean_G2}, FID = {FID_G2}, FIDmean = {FIDmean_G2}")
             writer.add_scalar("Validation IS Sample/G1(testset)-->(E2,G2)", IS[0],Counter_vald_epoch_batch*args.valbatches*args.OLbatchSize+cnt )
             writer.add_scalar("Validation FID Sample/G1(testset)-->(E2,G2)", FID_G2,Counter_vald_epoch_batch*args.valbatches*args.OLbatchSize+cnt )
            
             ## Validation by measuring Likelihood of G1
             Counter_G2test_E1 += 1
             sample_G2 = samples_G2test[cnt:cnt+args.OLbatchSize].view([-1,args.nc,args.imageSize,args.imageSize]).detach().to(device)
             _, logvar_first, z, _ = netE1Orig(sample_G2, args)
             likelihood_sample = engine_OGAN.get_likelihood(args,device,netE1,optimizerE1,sample_G2,netG1,logsigmaG1,args.ckptOL_E1,logvar_first)
             if (cnt == 0) and (Counter_G2test_E1 == 1):
                likelihood_G2test_E1 = likelihood_sample.detach().view(1,1)
                likelihood_G2test_E1_mean = likelihood_G2test_E1
             else:
                likelihood_G2test_E1 = torch.cat((likelihood_G2test_E1,likelihood_sample.detach().view(1,1)),1)
                #likelihood_G2test_E1_mean = torch.logsumexp(likelihood_G2test_E1,1)-torch.log(torch.tensor(likelihood_G2test_E1.shape[1]))
                likelihood_G2test_E1_mean = torch.mean(likelihood_G2test_E1,1)
             LL_G2test_E1_mean=likelihood_G2test_E1_mean
             #print(f"Validation: G2(testset)-->(E1,G1)({Counter_epoch_batch}): batch {Counter_G2test_E1} of {int(args.valbatches)}, LL (batch) = {likelihood_sample.item()}, LL (moving average) = {LL_G2test_E1_mean.item()}")
             writer.add_scalar("Validation LL Sample/G2(testset)-->(E1,G1)", likelihood_sample.item(),  Counter_vald_epoch_batch*args.valbatches*args.OLbatchSize+cnt )

             ## Measuring Inception Score
             x_hat = netG1(z)
             if x_hat.shape[1] != 3:
                x_hat = x_hat.repeat([1,3,1,1])
             IS = iscore.inception_score(x_hat, device, batch_size=32, resize=True, splits=1)
             ISsumG1=(ISsumG1+IS[0]);ISmean_G1=ISsumG1/Counter_G2test_E1

             ## Measuring FID score for sample_G2 (real) and x_hat=netG1(z) (fake)
             FID_G1 = fidscore.calculate_fretchet(sample_G2,x_hat,inception_model, device)
             FIDsumG1 = FIDsumG1 + FID_G1; FIDmean_G1=FIDsumG1/Counter_G2test_E1

             print(f"Validation: G2(testset)-->(E1,G1)({Counter_epoch_batch}): batch {Counter_G2test_E1} of {int(args.valbatches)}, LL (batch) = {likelihood_sample.item()}, LL (moving average) = {LL_G2test_E1_mean.item()} IS = {IS[0]}, ISmean = {ISmean_G1} , FID = {FID_G1}, FIDmean = {FIDmean_G1}")
             writer.add_scalar("Validation IS Sample/G2(testset)-->(E1,G1)", IS[0],Counter_vald_epoch_batch*args.valbatches*args.OLbatchSize+cnt )
             writer.add_scalar("Validation FID Sample/G2(testset)-->(E1,G1)", FID_G1,Counter_vald_epoch_batch*args.valbatches*args.OLbatchSize+cnt )  

             writer.add_scalar("Validation LL Moving Average/G1(testset)-->(E2,G2)", LL_G1test_E2_mean.item(), Counter_vald_epoch_batch*args.valbatches*args.OLbatchSize+cnt )
             writer.add_scalar("Validation LL Moving Average/G2(testset)-->(E1,G1)", LL_G2test_E1_mean.item(), Counter_vald_epoch_batch*args.valbatches*args.OLbatchSize+cnt )

             writer.add_scalar("IS Moving Average/G1(testset)-->(E2,G2)", ISmean_G2.item(), Counter_vald_epoch_batch*args.valbatches*args.OLbatchSize+cnt )
             writer.add_scalar("IS Moving Average/G2(testset)-->(E1,G1)", ISmean_G1.item(), Counter_vald_epoch_batch*args.valbatches*args.OLbatchSize+cnt )

             writer.add_scalar("FID Moving Average/G1(testset)-->(E2,G2)", FIDmean_G2.item(), Counter_vald_epoch_batch*args.valbatches*args.OLbatchSize+cnt )
             writer.add_scalar("FID Moving Average/G2(testset)-->(E1,G1)", FIDmean_G1.item(), Counter_vald_epoch_batch*args.valbatches*args.OLbatchSize+cnt )

         writer.add_scalar("Validation LL epoch/G2(testset)-->(E1,G1)", LL_G2test_E1_mean.item(), Counter_vald_epoch_batch )
         writer.add_scalar("Validation LL epoch/G1(testset)-->(E2,G2)", LL_G1test_E2_mean.item(), Counter_vald_epoch_batch )

         writer.add_scalar("IS epoch/G2(testset)-->(E1,G1)", ISmean_G2, Counter_vald_epoch_batch )
         writer.add_scalar("IS epoch/G1(testset)-->(E2,G2)", ISmean_G1, Counter_vald_epoch_batch )

         writer.add_scalar("FID epoch/G2(testset)-->(E1,G1)", FIDmean_G2, Counter_vald_epoch_batch )
         writer.add_scalar("FID epoch/G1(testset)-->(E2,G2)", FIDmean_G1, Counter_vald_epoch_batch )

    ##-- writing to Tensorboard
    if (args.mode == 'train') or (args.mode == 'train_validate'):
      if (Counter_epoch_batch % 5 == 0) or (Counter_epoch_batch % args.valevery == 0) or (Counter_epoch_batch == 1):
         writer.add_scalar("Overlap Loss_batch/OL[G2-->(E1,G1)]", OLossG1_No_W1, Counter_epoch_batch)
         writer.add_scalar("Overlap Loss_batch/OL[G1-->(E2,G2)]", OLossG2_No_W2, Counter_epoch_batch)
         writer.add_scalar("Overlap Loss_batch/OL[G2-->(E1,G1)] + OL[G1-->(E2,G2)]", TrueOLoss_No_W1W2, Counter_epoch_batch)
         writer.add_scalar("Overlap Loss_batch/ Distance(G1,G2)", Distance_G1G2_No_W, Counter_epoch_batch)
         writer.add_scalar("All Loss_batch/ Loss G1", AdvLossG1, Counter_epoch_batch)
         writer.add_scalar("All Loss_batch/ Loss G2", AdvLossG2, Counter_epoch_batch)

         DL_G1 = PresGANResultsG1[0]/Counter_epoch_batch
         GL_G1 = PresGANResultsG1[1]/Counter_epoch_batch
         Dx_G1 = PresGANResultsG1[2]/Counter_epoch_batch
         DL_G1_z1 = PresGANResultsG1[3]/Counter_epoch_batch
         DL_G1_z2 = PresGANResultsG1[4]/Counter_epoch_batch
         sigma_x_G1_min = PresGANResultsG1[5]/Counter_epoch_batch
         sigma_x_G1_max = PresGANResultsG1[6]/Counter_epoch_batch
         g_error_criterion_G1 = PresGANResultsG1[7]/Counter_epoch_batch
         g_error_entropy_G1 = PresGANResultsG1[8]/Counter_epoch_batch
         g_error_G1 = PresGANResultsG1[9]/Counter_epoch_batch


         DL_G2 = PresGANResultsG2[0]/Counter_epoch_batch
         GL_G2 = PresGANResultsG2[1]/Counter_epoch_batch
         Dx_G2 = PresGANResultsG2[2]/Counter_epoch_batch
         DL_G2_z1 = PresGANResultsG2[3]/Counter_epoch_batch
         DL_G2_z2 = PresGANResultsG2[4]/Counter_epoch_batch
         sigma_x_G2_min = PresGANResultsG2[5]/Counter_epoch_batch
         sigma_x_G2_max = PresGANResultsG2[6]/Counter_epoch_batch
         g_error_criterion_G2 = PresGANResultsG2[7]/Counter_epoch_batch
         g_error_entropy_G2 = PresGANResultsG2[8]/Counter_epoch_batch
         g_error_G2 = PresGANResultsG2[9]/Counter_epoch_batch


         writer.add_scalar("G1/G1-Loss/Loss_D", DL_G1, Counter_epoch_batch)
         writer.add_scalar("G1/G1-Loss/Loss_G", GL_G1, Counter_epoch_batch)
         writer.add_scalar("G1/G1-D(x)", Dx_G1, Counter_epoch_batch)
         writer.add_scalar("G1/G1-DL_G/DL_G_z1", DL_G1_z1, Counter_epoch_batch)
         writer.add_scalar("G1/G1-DL_G/DL_G_z2", DL_G1_z2, Counter_epoch_batch)
         writer.add_scalar("G1/G1-sigma/sigma_min", sigma_x_G1_min, Counter_epoch_batch)
         writer.add_scalar("G1/G1-sigma/sigma_max", sigma_x_G1_max, Counter_epoch_batch)
         writer.add_scalar("G1/G1-g_error_criterion_G1", g_error_criterion_G1, Counter_epoch_batch)
         writer.add_scalar("G1/G1-g_error_entropy_G1", g_error_entropy_G1, Counter_epoch_batch)
         writer.add_scalar("G1/G1-g_error_G1", g_error_G1, Counter_epoch_batch)

         writer.add_scalar("G2/G2-Loss/Loss_D", DL_G2, Counter_epoch_batch)
         writer.add_scalar("G2/G2-Loss/Loss_G", GL_G2, Counter_epoch_batch)
         writer.add_scalar("G2/G2-D(x)", Dx_G2, Counter_epoch_batch)
         writer.add_scalar("G2/G2-DL_G/DL_G_z1", DL_G2_z1, Counter_epoch_batch)
         writer.add_scalar("G2/G2-DL_G/DL_G_z2", DL_G2_z2, Counter_epoch_batch)
         writer.add_scalar("G2/G2-sigma/sigma_min", sigma_x_G2_min, Counter_epoch_batch)
         writer.add_scalar("G2/G2-sigma/sigma_max", sigma_x_G2_max, Counter_epoch_batch)
         writer.add_scalar("G2/G2-g_error_criterion_G2", g_error_criterion_G2, Counter_epoch_batch)
         writer.add_scalar("G2/G2-g_error_entropy_G2", g_error_entropy_G2, Counter_epoch_batch)
         writer.add_scalar("G2/G2-g_error_G2", g_error_G2, Counter_epoch_batch)


      if ((Counter_epoch_batch % int(len(trainsetG1)/args.batchSize) == 0)):
         writer.add_scalar("Overlap Loss_epoch/OL[G2-->(E1,G1)]", OLossG1_No_W1, epoch)
         writer.add_scalar("Overlap Loss_epoch/OL[G1-->(E2,G2)]", OLossG2_No_W2, epoch)
         writer.add_scalar("Overlap Loss_epoch/OL[G2-->(E1,G1)] + OL[G1-->(E2,G2)]", TrueOLoss_No_W1W2, epoch)
         writer.add_scalar("Overlap Loss_epoch/ Distance(G1,G2)", Distance_G1G2_No_W, epoch)
         writer.add_scalar("All Loss_epoch/ Loss G1", AdvLossG1, epoch)
         writer.add_scalar("All Loss_epoch/ Loss G2", AdvLossG2, epoch)

         writer.flush()

      ## save models
      if Counter_epoch_batch % 50 ==0:
         torch.save(netG1.state_dict(), os.path.join(args.ckptOL_G, 'netG1_presgan_%s_step_%s.pth'%(args.dataset, Counter_epoch_batch)))
         torch.save(logsigmaG1, os.path.join(args.ckptOL_G, 'log_sigma_G1_%s_step_%s.pth'%(args.dataset, Counter_epoch_batch)))
         torch.save(netD1.state_dict(), os.path.join(args.ckptOL_G, 'netD1_presgan_%s_step_%s.pth'%(args.dataset, Counter_epoch_batch)))

         torch.save(netG2.state_dict(), os.path.join(args.ckptOL_G, 'netG2_presgan_%s_step_%s.pth'%(args.dataset, Counter_epoch_batch)))
         torch.save(logsigmaG2, os.path.join(args.ckptOL_G, 'log_sigma_G2_%s_step_%s.pth'%(args.dataset, Counter_epoch_batch)))
         torch.save(netD2.state_dict(), os.path.join(args.ckptOL_G, 'netD2_presgan_%s_step_%s.pth'%(args.dataset, Counter_epoch_batch)))

