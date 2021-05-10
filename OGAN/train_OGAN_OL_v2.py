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
   pdb.set_trace()

 #logsigma_init = -1 #initial value for log_sigma_sian
 if logsigma_file != '':
    logsigmaG = torch.load(logsigma_file)
 else:
   print('A valid logsigma_file for a pretrained PGAN generator must be provided')
   pdb.set_trace()
 return netG, logsigmaG

##-- loading PGAN discriminator model
def load_discriminator(netD,device,ckptD):
 netD.apply(utils.weights_init)
 if ckptD != '':
    netD.load_state_dict(torch.load(ckptD))
 else:
   print('A valid ckptD for a pretrained PGAN discriminator must be provided')
   pdb.set_trace()
 return netD


if __name__ == "__main__":
 ##-- run on the available GPU otherwise CPUs
 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 ##-- preparing folders to save results of Likelihood
 OL_folders(args)

 ##-- loading and spliting datasets
 trainset, testset = load_datasets(data,args,device)


 ##-- loading PGAN generator model with sigma and setting generator training parameters - G1
 netG = nets.Generator(args).to(device)
 optimizerG = optim.Adam(netG.parameters(), lr=args.lrG1)
 netG, log_sigma = load_generator_wsigma(netG,device,args.ckptG1,args.logsigma_file_G1)
 sigma_optimizer = optim.Adam([log_sigma], lr=args.sigma_lr, betas=(args.beta1, 0.999))


 ##-- loading PGAN discriminator model and setting discriminator training parameters - D1
 netD = nets.Discriminator(args).to(device)
 netD = load_discriminator(netD,device,args.ckptD1)
 optimizerD = optim.Adam(netD.parameters(), lr=args.lrD1, betas=(args.beta1, 0.999))

 if True:
#### defining generator
    #netG = nets.Generator(args).to(device)
    #log_sigma = torch.tensor([args.logsigma_init]*(args.imageSize*args.imageSize), device=device, requires_grad=True)

    #### defining discriminator
    #netD = nets.Discriminator(args).to(device) 

    #### initialize weights
    #netG.apply(utils.weights_init)
    #if args.ckptG1 != '':
    #    netG.load_state_dict(torch.load(args.ckptG1))

    #netD.apply(utils.weights_init)
    #if args.ckptD1 != '':
    #    netD.load_state_dict(torch.load(args.ckptD1))
    #    pdb.set_trace()

    #if args.logsigma_file_G1 != '':
    #    log_sigma = torch.load(args.logsigma_file_G1)

    writer = SummaryWriter(args.ckptOL_G1I)
    #device = args.device
    X_training = trainset.to(device)
    fixed_noise = torch.randn(args.num_gen_images, args.nzg, 1, 1, device=device)
    #optimizerD = optim.Adam(netD.parameters(), lr=args.lrD1, betas=(args.beta1, 0.999))
    #optimizerG = optim.Adam(netG.parameters(), lr=args.lrG1, betas=(args.beta1, 0.999)) 
    #sigma_optimizer = optim.Adam([log_sigma], lr=args.sigma_lr, betas=(args.beta1, 0.999))
    if args.restrict_sigma:
        logsigma_min = math.log(math.exp(args.sigma_min) - 1.0)
        logsigma_max = math.log(math.exp(args.sigma_max) - 1.0)
    stepsize = args.stepsize_num / args.nzg
    
    bsz = args.batchSize
    for epoch in range(1, args.epochs+1):
        DL=0
        GL=0
        Dx=0
        DL_G_z1=0
        DL_G_z2=0
        Counter = 0
        for i in range(0, len(X_training), bsz): 
            Counter = Counter+1
            sigma_x = F.softplus(log_sigma).view(1, 1, args.imageSize, args.imageSize).to(device)

            netD.zero_grad()
            stop = min(bsz, len(X_training[i:]))
            real_cpu = X_training[i:i+stop].to(device)

            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device, dtype=torch.float32)

            noise_eta = torch.randn_like(real_cpu)
            noised_data = real_cpu + sigma_x.detach() * noise_eta
            out_real = netD(noised_data)
            errD_real = criterion(out_real, label)
            errD_real.backward()
            D_x = out_real.mean().item()

            # train with fake
            
            noise = torch.randn(batch_size, args.nzg, 1, 1, device=device)
            mu_fake = netG(noise) 
            fake = mu_fake + sigma_x * noise_eta
            label.fill_(fake_label)
            out_fake = netD(fake.detach())
            errD_fake = criterion(out_fake, label)
            errD_fake.backward()
            D_G_z1 = out_fake.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # update G network: maximize log(D(G(z)))

            netG.zero_grad()
            sigma_optimizer.zero_grad()

            label.fill_(real_label)  
            gen_input = torch.randn(batch_size, args.nzg, 1, 1, device=device)
            out = netG(gen_input)
            noise_eta = torch.randn_like(out)
            g_fake_data = out + noise_eta * sigma_x

            dg_fake_decision = netD(g_fake_data)
            g_error_gan = criterion(dg_fake_decision, label) 
            D_G_z2 = dg_fake_decision.mean().item()

            if args.lambda_ == 0:
                g_error_gan.backward()
                optimizerG.step() 
                sigma_optimizer.step()

            else:
                hmc_samples, acceptRate, stepsize = hmc.get_samples(
                    netG, g_fake_data.detach(), gen_input.clone(), sigma_x.detach(), args.burn_in, 
                        args.num_samples_posterior, args.leapfrog_steps, stepsize, args.flag_adapt, 
                            args.hmc_learning_rate, args.hmc_opt_accept)
                
                bsz, d = hmc_samples.size()
                mean_output = netG(hmc_samples.view(bsz, d, 1, 1).to(device))
                bsz = g_fake_data.size(0)

                mean_output_summed = torch.zeros_like(g_fake_data)
                for cnt in range(args.num_samples_posterior):
                    mean_output_summed = mean_output_summed + mean_output[cnt*bsz:(cnt+1)*bsz]
                mean_output_summed = mean_output_summed / args.num_samples_posterior  

                c = ((g_fake_data - mean_output_summed) / sigma_x**2).detach()
                g_error_entropy = torch.mul(c, out + sigma_x * noise_eta).mean(0).sum()

                g_error = g_error_gan - args.lambda_ * g_error_entropy
                g_error.backward()
                optimizerG.step() 
                sigma_optimizer.step()

            if args.restrict_sigma:
                log_sigma.data.clamp_(min=logsigma_min, max=logsigma_max)

            ## log performance
            if i % args.log == 0:
                print('Epoch [%d/%d] .. Batch [%d/%d] .. Loss_D: %.4f .. Loss_G: %.4f .. D(x): %.4f .. D(G(z)): %.4f / %.4f'
                        % (epoch, args.epochs, i, len(X_training), errD.data, g_error_gan.data, D_x, D_G_z1, D_G_z2))

            DL = DL + errD.data
            GL = GL + g_error_gan.data
            Dx = Dx + D_x
            DL_G_z1 = DL_G_z1 + D_G_z1
            DL_G_z2 = DL_G_z2 + D_G_z2

            if Counter % 10 == 0:
              fake = netG(fixed_noise).detach()
              # log images to tensorboard
              # create grid of images
              img_grid = torchvision.utils.make_grid(fake)

              # write to tensorboard
              writer.add_image('fake_images', img_grid, Counter)
              # --------------



        DL = DL/Counter
        GL = GL/Counter
        Dx = Dx/Counter
        DL_G_z1 = DL_G_z1/Counter
        DL_G_z2 = DL_G_z2/Counter

        #log performance to tensorboard
        writer.add_scalar("Loss/Loss_D", DL, epoch)
        writer.add_scalar("Loss/Loss_G", GL, epoch) 
        writer.add_scalar("D(x)", Dx, epoch) 
        writer.add_scalar("DL_G/DL_G_z1", DL_G_z1, epoch) 
        writer.add_scalar("DL_G/DL_G_z2", DL_G_z2, epoch) 
        writer.add_scalar("sigma/sigma_min", torch.min(sigma_x), epoch) 
        writer.add_scalar("sigma/sigma_max", torch.max(sigma_x), epoch) 
        #----------------
        #pdb.set_trace()

        print('*'*100)
        print('End of epoch {}'.format(epoch))
        print('Epoch [%d/%d] .. Loss_D: %.4f .. Loss_G: %.4f .. D(x): %.4f .. D(G(z)): %.4f / %.4f'
                        % (epoch, args.epochs, DL, GL, Dx, DL_G_z1, DL_G_z2))

        print('sigma min: {} .. sigma max: {}'.format(torch.min(sigma_x), torch.max(sigma_x)))
        print('*'*100)
        if args.lambda_ > 0:
            print('| MCMC diagnostics ====> | stepsize: {} | min ar: {} | mean ar: {} | max ar: {} |'.format(
                        stepsize, acceptRate.min().item(), acceptRate.mean().item(), acceptRate.max().item()))

        if epoch % args.save_imgs_every == 0:
            fake = netG(fixed_noise).detach()
            vutils.save_image(fake, '%s/presgan_%s_fake_epoch_%03d.png' % (args.ckptOL_G1I, args.dataset, epoch), normalize=True, nrow=20) 

            # log images to tensorboard
            # create grid of images
            #img_grid = torchvision.utils.make_grid(fake)

            # write to tensorboard
            #writer.add_image('fake_images', img_grid, epoch)
            # --------------


        if epoch % args.save_imgs_every == 0:
            torch.save(netG.state_dict(), os.path.join(args.ckptOL_G1I, 'netG_presgan_%s_epoch_%s.pth'%(args.dataset, epoch)))
            torch.save(log_sigma, os.path.join(args.ckptOL_G1I, 'log_sigma_%s_%s.pth'%(args.dataset, epoch)))
            torch.save(netD.state_dict(), os.path.join(args.ckptOL_G1I, 'netD_presgan_%s_epoch_%s.pth'%(args.dataset, epoch)))
    
    writer.flush()
