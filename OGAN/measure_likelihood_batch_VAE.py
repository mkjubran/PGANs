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
import pickle

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

if __name__ == "__main__":
 ##-- run on the available GPU otherwise CPUs
 device = torch.device('cuda:'+str(args.GPU) if torch.cuda.is_available() else 'cpu')

 ##-- preparing folders to save results of Likelihood
 likelihood_folders(args)

 ##-- loading and spliting datasets
 trainset, testset = load_datasets(data,args,device, args.seed_VAE)

 ## -- setting seeds for all the rest of the code
 torch.manual_seed(0)
 random.seed(0)
 np.random.seed(0)

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

 log_dir = args.save_likelihood_folder
 #log_dir = args.save_likelihood_folder+"/LResults_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
 writer = SummaryWriter(log_dir)

 Counter = 0
 Counter_G1test_E2 = 0
 likelihood_Dec_train = []
 likelihood_Dec_test = []

 MinNTrain = min(trainset.shape[0],args.number_samples_likelihood)
 MinNTest = min(testset.shape[0],args.number_samples_likelihood)
 args.number_samples_likelihood = MinNTrain

 #samples_G1 = trainset[random.sample(range(0, len(trainsetG1)), MinNTrain)]
 #samples_G1test = testset[random.sample(range(0, len(testset)), MinNTest)]
 samples_G1 = trainset[0:MinNTrain]
 samples_G1test = testset[0:MinNTest]


 for j in range(0, args.number_samples_likelihood, args.OLbatchSize):
    Counter += 1

    netDec.eval()
    ##-- compute OL where samples from G1 are applied to (E2,G2)
    sample_G1 = samples_G1[j:j+args.OLbatchSize].view([-1,1,args.imageSize,args.imageSize]).detach().to(device)
    likelihood_samples = engine_OGAN.get_likelihood_VAE(args,device,netE,optimizerE,sample_G1,netDec,args.save_likelihood_folder)
    likelihood_sample = torch.mean(likelihood_samples,0)
    likelihood_Dec_train.extend(likelihood_samples.tolist())
    print(f"Dataset (train)-->VAE Dec: batch {Counter} of {int(args.number_samples_likelihood/args.OLbatchSize)}, OL = {likelihood_sample.item()}, moving mean = {statistics.mean(likelihood_Dec_train)}")
    writer.add_scalar("Validation LL/Sample: Dataset(train)-->VAE Dec", likelihood_sample.item(), Counter)
    writer.add_scalar("Validation LL/Moving Average: Dataset(train)-->VAE Dec", statistics.mean(likelihood_Dec_train), Counter)

    ##-- validation step
    if (j+args.OLbatchSize <= samples_G1test.shape[0]):
         netDec.eval()

         ## Validation by measuring Likelihood of VAE
         Counter_G1test_E2 += 1
         sample_G1 = samples_G1test[j:j+args.OLbatchSize].view([-1,1,args.imageSize,args.imageSize]).detach().to(device)
         likelihood_samples = engine_OGAN.get_likelihood_VAE(args,device,netE,optimizerE,sample_G1,netDec,args.save_likelihood_folder)
         likelihood_sample = torch.mean(likelihood_samples,0)
         likelihood_Dec_test.extend(likelihood_samples.tolist())
         print(f"Validation: G1(testset)--> VAE Dec: batch {Counter_G1test_E2} of {int(args.valbatches)}, OL = {likelihood_sample.item()}, moving mean = {statistics.mean(likelihood_Dec_test)}")
         writer.add_scalar("Validation LL/Sample: Dataset(test)-->VAE Dec", likelihood_sample.item(), Counter)
         writer.add_scalar("Validation LL/Moving Average: Dataset(test)-->VAE Dec", statistics.mean(likelihood_Dec_test), Counter)

         if (j % 10 == 0) or (j+2*args.OLbatchSize > samples_G1test.shape[0]) :
           with open(args.save_likelihood_folder+'/likelihood_Dec_test.pkl', 'wb') as f:
              pickle.dump(likelihood_Dec_test, f)

    if ( j % 10 ==0 ) or (j + 2*args.OLbatchSize > args.number_samples_likelihood):
        with open(args.save_likelihood_folder+'/likelihood_Dec_train.pkl', 'wb') as f:
           pickle.dump(likelihood_Dec_train, f)

 writer.flush()
 writer.close()
