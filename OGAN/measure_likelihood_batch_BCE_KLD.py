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
import datetime
import random
import sys
import math
import pickle

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

parser.add_argument('--sample_from', type=str, default='generator', help='Sample from generator | dataset')
parser.add_argument('--save_likelihood_folder', type=str, default='../../outputs', help='where to save generated images')
parser.add_argument('--number_samples_likelihood', type=int, default=100, help='Number of Samples to considered to compute Likelihood')

parser.add_argument('--lrOL', type=float, default=0.001, help='learning rate for overlap loss, default=0.001')
parser.add_argument('--OLbatchSize', type=int, default=100, help='Overlap Loss batch size')
parser.add_argument('--S', type=int, default=1000, help='Sample Size when computing Likelihood')
parser.add_argument('--overdispersion', type=float, default=1.2, help='overdispersion parameter')

parser.add_argument('--dataset', required=True, help=' ring | mnist | stackedmnist | cifar10 ')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--OLepochs', type=int, default=1000, help='number of epochs to train for Overlap Loss')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--overlap_loss_min', type=int, default=-50000, help='min value for Overlap Loss in VAE to determine the proposal')
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

parser.add_argument('--GPU', type=int, default=0, help='GPU to use')

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
 mvn = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(scale).reshape(1, imageSize*imageSize, imageSize*imageSize), validate_args = False)
 log_pxz_mvn = mvn.log_prob(data.view(-1,imageSize*imageSize))

 ##-- computer sample from standard normal distribution
 std = torch.exp(0.5*logvar)
 std_b = torch.eye(std.size(1)).to(device)
 std_c = std.unsqueeze(2).expand(*std.size(), std.size(1))
 std_3d = std_c * std_b
 mvnz = torch.distributions.MultivariateNormal(mu, scale_tril=std_3d, validate_args = False)
 pz_normal = torch.exp(mvnz.log_prob(zr))
 return log_pxz_mvn, pz_normal

##-- Sample from Generator
def sample_from_generator(args,netG ,NoSamples):
 ##-- sample from standard normal distribution
 nz=100
 mean = torch.zeros(NoSamples,nz).to(device)
 scale = torch.ones(nz).to(device)
 mvn = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(scale).view(1, nz, nz), validate_args = False)
 sample_z_shape = torch.Size([])
 sample_z = mvn.sample(sample_z_shape).view(-1,nz,1,1)
 recon_images = netG(sample_z)
 return recon_images

if __name__ == "__main__":
 ##-- run on the available GPU otherwise CPUs
 device = torch.device('cuda:'+str(args.GPU) if torch.cuda.is_available() else 'cpu')

 ##-- preparing folders to save results of Likelihood
 likelihood_folders(args)

 ##-- loading and spliting datasets for G1
 trainsetG1, testsetG1 = load_datasets(data,args,device, args.seed_G1)

 ##-- loading and spliting datasets for G2
 trainsetG2, testsetG2 = load_datasets(data,args,device, args.seed_G2)

 ## -- setting seeds for all the rest of the code
 torch.manual_seed(0)
 random.seed(0)
 np.random.seed(0)

 ##-- loading PGAN generator model with sigma and setting generator training parameters - G1
 netG1 = nets.GeneratorSigmoid(args).to(device)
 netG1, logsigmaG1 = load_generator_wsigma(netG1,device,args.ckptG1,args.logsigma_file_G1)
 optimizerG1 = optim.Adam(netG1.parameters(), lr=args.lrG1, betas=(args.beta1, 0.999))
 sigma_optimizerG1 = optim.Adam([logsigmaG1], lr=args.sigma_lr, betas=(args.beta1, 0.999))

 ##-- loading PGAN generator model with sigma and setting generator training parameters - G2
 netG2 = nets.GeneratorSigmoid(args).to(device)
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

 ##-- setting scale and selecting a random test sample
 scale = 0.01*torch.ones(args.imageSize**2)
 scale = scale.to(device)
 #i = torch.randint(0, len(testset),(1,1)) ## selection of the index of test image

 ##-- define a new encoder netES to find OL per sample (need to keep the orogonal netE))
 netES = nets.ConvVAEType2(args).to(device)
 optimizerES = optim.Adam(netES.parameters(), lr=args.lrOL)

 testsetG1= testsetG1 #.to(device)
 trainsetG1= trainsetG1 #.to(device)

 testsetG2= testsetG2 #.to(device)
 trainsetG2= trainsetG2 #.to(device)

 #log_dir = args.save_likelihood_folder+"/LResults_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
 log_dir = args.save_likelihood_folder
 writer = SummaryWriter(log_dir)

 Counter_G1_E2 = 0
 Counter_G2_E1 = 0
 Counter_G1test_E2 = 0
 Counter_G2test_E1 = 0
 likelihood_G1_E2 = []
 likelihood_G2_E1 = []
 if args.sample_from == 'generator':
    samples_G1 = sample_from_generator(args, netG1, args.number_samples_likelihood)
    samples_G2 = sample_from_generator(args, netG2, args.number_samples_likelihood)
 elif args.sample_from == 'dataset':

    MinNTrain = min(trainsetG1.shape[0],args.number_samples_likelihood)
    MinNTest = min(testsetG1.shape[0],args.number_samples_likelihood)

    #samples_G1 = trainsetG1[random.sample(range(0, len(trainsetG1)), MinNTrain)]
    #samples_G2 = trainsetG2[random.sample(range(0, len(trainsetG2)), MinNTrain)]

    likelihood_G1test_E2 = []
    likelihood_G2test_E1 = []
    #samples_G1test = testsetG1[random.sample(range(0, len(testsetG1)), MinNTest)] 
    #samples_G2test = testsetG2[random.sample(range(0, len(testsetG2)), MinNTest)] 

    samples_G1 = trainsetG1[0:MinNTrain]
    samples_G2 = trainsetG2[0:MinNTrain]
    samples_G1test = testsetG1[0:MinNTest]
    samples_G2test = testsetG2[0:MinNTest]

 else:
    print('Can not sample from {}. Sample from should be either generator or dataset!!!'.format(args.sample_from))
    sys.exit(1)

 for j in range(0, args.number_samples_likelihood, args.OLbatchSize):
    Counter_G1_E2 += 1
    Counter_G2_E1 += 1
    Counter_G1test_E2 += 1
    Counter_G2test_E1 += 1

    ##-- compute OL where samples from G1 are applied to (E2,G2)
    netE2.load_state_dict(copy.deepcopy(netE2Orig.state_dict()))
    optimizerE2 = optim.Adam(netE2.parameters(), lr=args.lrE2)
    netE2.train()
    sample_G1 = samples_G1[j:j+args.OLbatchSize].view([-1,args.nc,args.imageSize,args.imageSize]).detach().to(device)
    _, logvar_first, _, _ = netE2Orig(sample_G1, args)
    #likelihood_samples = engine_OGAN.get_likelihood_MLL(args,device,netE2,optimizerE2,sample_G1,netG2,logsigmaG2,args.save_likelihood_folder,logvar_first)
    #likelihood_sample = torch.mean(likelihood_samples,0)

    likelihood_samples = engine_OGAN.get_likelihood_MLL_BCE_KLD(args,device,netE2,optimizerE2,sample_G1,netG2,args.save_likelihood_folder,logvar_first)
    likelihood_sample = torch.mean(likelihood_samples,0)

    if math.isnan(likelihood_sample): ## to avoid training generator with nan loss
      print(f"G1-->(E2,G2): batch {Counter_G1_E2} of {int(args.number_samples_likelihood/args.OLbatchSize)}, Likelihood = {likelihood_sample.item()}")
      Counter_G1_E2 -= 1
    else:
      #likelihood_G1_E2.extend(likelihood_samples.tolist())
      likelihood_G1_E2.append(likelihood_samples.tolist())
      if Counter_G1_E2 % 1 == 0:
         print(f"G1-->(E2,G2): batch {Counter_G1_E2} of {int(args.number_samples_likelihood/args.OLbatchSize)}, OL = {likelihood_sample.item()}, moving mean = {statistics.mean(likelihood_G1_E2)}")
      writer.add_scalar("Measure LL/Batch: G1-->(E2,G2)", likelihood_sample.item(), Counter_G1_E2)
      writer.add_scalar("Measure LL/Moving Average: G1-->(E2,G2)", statistics.mean(likelihood_G1_E2), Counter_G1_E2)


    ##-- compute OL where samples from G2 are applied to (E1,G1)
    netE1.load_state_dict(copy.deepcopy(netE1Orig.state_dict()))
    optimizerE1 = optim.Adam(netE1.parameters(), lr=args.lrE1)
    netE1.train()
    sample_G2 = samples_G2[j:j+args.OLbatchSize].view([-1,args.nc,args.imageSize,args.imageSize]).detach().to(device)
    _, logvar_first, _, _ = netE1Orig(sample_G2, args)
    #likelihood_samples = engine_OGAN.get_likelihood_MLL(args,device,netE1,optimizerE1,sample_G2,netG1,logsigmaG1,args.save_likelihood_folder,logvar_first)
    #likelihood_sample = torch.mean(likelihood_samples,0)

    likelihood_samples = engine_OGAN.get_likelihood_MLL_BCE_KLD(args,device,netE1,optimizerE1,sample_G2,netG1,args.save_likelihood_folder,logvar_first)
    likelihood_sample = torch.mean(likelihood_samples,0)

    if math.isnan(likelihood_sample): ## to avoid training generator with nan loss
      print(f"G2-->(E1,G1): batch {Counter_G2_E1} of {int(args.number_samples_likelihood/args.OLbatchSize)}, Likelihood = {likelihood_sample.item()}")
      Counter_G2_E1 -= 1
    else:
      #likelihood_G2_E1.extend(likelihood_samples.tolist())
      likelihood_G2_E1.append(likelihood_samples.tolist())
      if Counter_G2_E1 % 1 == 0:
         print(f"G2-->(E1,G1): batch {Counter_G2_E1} of {int(args.number_samples_likelihood/args.OLbatchSize)}, OL = {likelihood_sample.item()}, moving mean = {statistics.mean(likelihood_G2_E1)}")
      writer.add_scalar("Measure LL/Bacth: G2-->(E1,G1)", likelihood_sample.item(), Counter_G2_E1)
      writer.add_scalar("Measure LL/Moving Average: G2-->(E1,G1)", statistics.mean(likelihood_G2_E1), Counter_G2_E1)

      ##-- compute Average L for test samples
      likelihood_G1_G2=torch.cat((torch.FloatTensor(likelihood_G2_E1).view(-1,1),torch.FloatTensor(likelihood_G1_E2).view(-1,1)),1)
      AvgLL=torch.mean(torch.add(torch.logsumexp(likelihood_G1_G2,1),-1*math.log(2)))
      if Counter_G2_E1 % 10 == 0:
           print(f"Moving Average: batch {Counter_G2_E1} of {int(args.number_samples_likelihood/args.OLbatchSize)}, log(0.5L(G1(random))+0.5L(G2(random))) = {AvgLL.item()}")
      writer.add_scalar("Measure LL/ Moving Average: log(0.5L(G1(random))+0.5L(G2(random)))", AvgLL.item(), Counter_G2test_E1)

    if args.sample_from == 'dataset' and (samples_G1test.shape[0] >= j+args.OLbatchSize):
      ##-- compute OL where samples from G1(testset) are applied to (E2,G2)
      netE2.load_state_dict(copy.deepcopy(netE2Orig.state_dict()))
      optimizerE2 = optim.Adam(netE2.parameters(), lr=args.lrE2)
      netE2.train()
      sample_G1 = samples_G1test[j:j+args.OLbatchSize].view([-1,args.nc,args.imageSize,args.imageSize]).detach().to(device)
      _, logvar_first, _, _ = netE2Orig(sample_G1, args)
      #likelihood_samples = engine_OGAN.get_likelihood_MLL(args,device,netE2,optimizerE2,sample_G1,netG2,logsigmaG2,args.save_likelihood_folder,logvar_first)
      #likelihood_sample = torch.mean(likelihood_samples,0)

      likelihood_samples = engine_OGAN.get_likelihood_MLL_BCE_KLD(args,device,netE2,optimizerE2,sample_G1,netG2,args.save_likelihood_folder,logvar_first)
      likelihood_sample = torch.mean(likelihood_samples,0)

      if math.isnan(likelihood_sample): ## to avoid training generator with nan loss
        print(f"G1(testset)-->(E2,G2): batch {Counter_G1test_E2} of {int(args.number_samples_likelihood/args.OLbatchSize)}, Likelihood = {likelihood_sample.item()}")
        Counter_G1test_E2 -= 1
      else:
        #likelihood_G1test_E2.extend(likelihood_samples.tolist())
        likelihood_G1test_E2.append(likelihood_samples.tolist())
        if Counter_G1test_E2 % 1 == 0:
           print(f"G1(testset)-->(E2,G2): batch {Counter_G1test_E2} of {int(args.number_samples_likelihood/args.OLbatchSize)}, OL = {likelihood_sample.item()}, moving mean = {statistics.mean(likelihood_G1test_E2)}")
        writer.add_scalar("Measure LL/Batch: Testset-->(E2,G2)", likelihood_sample.item(), Counter_G1test_E2)
        writer.add_scalar("Measure LL/Moving Average: G1(testset)-->(E2,G2)", statistics.mean(likelihood_G1test_E2), Counter_G1test_E2)

      ##-- compute OL where samples from G2(testset) are applied to (E1,G1)
      netE1.load_state_dict(copy.deepcopy(netE1Orig.state_dict()))
      optimizerE1 = optim.Adam(netE1.parameters(), lr=args.lrE1)
      netE1.train()
      sample_G2 = samples_G2test[j:j+args.OLbatchSize].view([-1,args.nc,args.imageSize,args.imageSize]).detach().to(device)
      _, logvar_first, _, _ = netE1Orig(sample_G2, args)
      #likelihood_samples = engine_OGAN.get_likelihood_MLL(args,device,netE1,optimizerE1,sample_G2,netG1,logsigmaG1,args.save_likelihood_folder,logvar_first)
      #likelihood_sample = torch.mean(likelihood_samples,0)

      likelihood_samples = engine_OGAN.get_likelihood_MLL_BCE_KLD(args,device,netE1,optimizerE1,sample_G2,netG1,args.save_likelihood_folder,logvar_first)
      likelihood_sample = torch.mean(likelihood_samples,0)

      if math.isnan(likelihood_sample): ## to avoid training generator with nan loss
        print(f"G2(testset)-->(E1,G1): batch {Counter_G2test_E1} of {int(args.number_samples_likelihood/args.OLbatchSize)}, Likelihood = {likelihood_sample.item()}")
        Counter_G2test_E1 -= 1
      else:
        #likelihood_G2test_E1.extend(likelihood_samples.tolist())
        likelihood_G2test_E1.append(likelihood_samples.tolist())
        if Counter_G2test_E1 % 1 == 0:
           print(f"G2(testset)-->(E1,G1): batch {Counter_G2test_E1} of {int(args.number_samples_likelihood/args.OLbatchSize)}, OL = {likelihood_sample.item()}, moving mean = {statistics.mean(likelihood_G2test_E1)}")
        writer.add_scalar("Measure LL/Batch: Testset-->(E1,G1)", likelihood_sample.item(), Counter_G2test_E1)
        writer.add_scalar("Measure LL/Moving Average: G2(testset)-->(E1,G1)", statistics.mean(likelihood_G2test_E1), Counter_G2test_E1)

        ##-- compute Average L for test samples
        #pdb.set_trace()
        likelihood_G1_G2=torch.cat((torch.FloatTensor(likelihood_G2test_E1).view(-1,1),torch.FloatTensor(likelihood_G1test_E2).view(-1,1)),1)
        AvgLL=torch.mean(torch.add(torch.logsumexp(likelihood_G1_G2,1),-1*math.log(2)))
        if Counter_G2test_E1 % 10 == 0:
           print(f"Moving Average: batch {Counter_G2test_E1} of {int(args.number_samples_likelihood/args.OLbatchSize)}, log(0.5L(G1(test))+0.5L(G2(test))) = {AvgLL.item()}")
        writer.add_scalar("Measure LL/ Moving Average: log(0.5L(G1(test))+0.5L(G2(test)))", AvgLL.item(), Counter_G2test_E1)

        if ( j % 10 ==0 ) or (j + args.OLbatchSize > args.number_samples_likelihood):
           with open(args.save_likelihood_folder+'/likelihood_G2_E1.pkl', 'wb') as f:
              pickle.dump(likelihood_G2_E1, f)

           with open(args.save_likelihood_folder+'/likelihood_G1_E2.pkl', 'wb') as f:
              pickle.dump(likelihood_G1_E2, f)

           with open(args.save_likelihood_folder+'/likelihood_G2test_E1.pkl', 'wb') as f:
              pickle.dump(likelihood_G2test_E1, f)

           with open(args.save_likelihood_folder+'/likelihood_G1test_E2.pkl', 'wb') as f:
              pickle.dump(likelihood_G1test_E2, f)


 writer.flush()
 writer.close()
