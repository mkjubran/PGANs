import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.utils as vutils
import torchvision

from torch.utils.tensorboard import SummaryWriter #Jubran

import seaborn as sns
import os 
import pickle 
import math 

import utils 
import hmc 

import pdb

from torch.distributions.normal import Normal

real_label = 1
fake_label = 0
criterion = nn.BCELoss()
criterion_mse = nn.MSELoss()


def presgan_encoder(dat, netG, netE, args):
    writer = SummaryWriter(args.results_folder_TB)
    device = args.device
    X_training = dat['X_train'].to(device)
    fixed_noise = torch.randn(args.num_gen_images, args.nz, 1, 1, device=device)
    optimizerE = optim.Adam(netE.parameters(), lr=args.lrD, betas=(args.beta1, 0.999))
    #optimizerG = optim.Adam(netG.parameters(), lr=args.lrG, betas=(args.beta1, 0.999)) 
    for epoch in range(1, args.epochs+1):
        for i in range(0, len(X_training), args.batchSize):
            netE.zero_grad()
            stop = min(args.batchSize, len(X_training[i:]))
            real_cpu = X_training[i:i+stop].to(device)

            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device, dtype=torch.int8)

            outputE = netE(real_cpu)
            outputG = netG(outputE)


            errE = criterion_mse(real_cpu, outputG)

            errE.backward()
            E_x = outputG.mean().item()
            optimizerE.step()

            ## log performance
            if i % args.log == 0:
                print('Epoch [%d/%d] .. Batch [%d/%d] .. Loss_E: %.4f .. E(x): %.4f'
                        % (epoch, args.epochs, i, len(X_training), errE.data, E_x))

                #log performance to tensorboard
                writer.add_scalar("Loss_E/train", errE.data, epoch)
                writer.add_scalar("E_x/train", E_x, epoch)
                #-------------

        print('*'*100)
        print('End of epoch {}'.format(epoch))
        print('*'*100)

        if epoch % args.save_ckpt_every == 0:
            torch.save(netG.state_dict(), os.path.join(args.results_folder, 'netG_dcgan_%s_epoch_%s.pth'%(args.dataset, epoch)))

    writer.flush()


def presgan_vaencoder(dat, netG, netE, args):
    writer = SummaryWriter(args.results_folder_TB)
    device = args.device
    X_training = dat['X_train'].to(device)
    fixed_noise = torch.randn(args.num_gen_images, args.nz, 1, 1, device=device)
    optimizerE = optim.Adam(netE.parameters(), lr=args.lrD, betas=(args.beta1, 0.999))
    #optimizerG = optim.Adam(netG.parameters(), lr=args.lrG, betas=(args.beta1, 0.999)) 
    for epoch in range(1, args.epochs+1):
        for i in range(0, len(X_training), args.batchSize):
            netE.zero_grad()
            stop = min(args.batchSize, len(X_training[i:]))
            real_cpu = X_training[i:i+stop].to(device)

            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device, dtype=torch.int8)

            mu, log_var = netE(real_cpu)
            #pdb.set_trace()

            # sample z from q
            std = torch.exp(log_var / 2)
            q = torch.distributions.Normal(mu, std)
            z = q.rsample()
            pdb.set_trace()

            # decoded - GAN Generator
            outputG = netG(z)
            pdb.set_trace()

            # reconstruction loss
            #recon_loss = self.gaussian_likelihood(outputG, self.log_scale, real_cpu)
            #pdb.set_trace()

            # kl
            #kl = self.kl_divergence(z, mu, std)
            #pdb.set_trace()

            # elbo
            #elbo = (kl - recon_loss)
            #pdb.set_trace()

            #elbo = elbo.mean()
            #pdb.set_trace()

            #errE = criterion_mse(real_cpu, outputG,args)

            #errE.backward()
            #E_x = outputG.mean().item()
            #optimizerE.step()

            ## log performance
            if i % args.log == 0:
                print('Epoch [%d/%d] .. Batch [%d/%d] .. Loss_E: %.4f .. E(x): %.4f'
                        % (epoch, args.epochs, i, len(X_training), errE.data, E_x))

                #log performance to tensorboard
                writer.add_scalar("Loss_E/train", errE.data, epoch)
                writer.add_scalar("E_x/train", E_x, epoch)
                #-------------

        print('*'*100)
        print('End of epoch {}'.format(epoch))
        print('*'*100)

        if epoch % args.save_ckpt_every == 0:
            torch.save(netG.state_dict(), os.path.join(args.results_folder, 'netG_dcgan_%s_epoch_%s.pth'%(args.dataset, epoch)))

    writer.flush()
