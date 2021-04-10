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

import PGAN_VAE

from torch.distributions.normal import Normal

real_label = 1
fake_label = 0
criterion = nn.BCELoss()
criterion_mse = nn.MSELoss()

def train_PGAN_VAE(dat, netG, args):
    writer = SummaryWriter(args.results_folder_TB)
    device = args.device

    VAE = PGAN_VAE.Network(dat, netG, args.nz, args.ngf, dat['nc'])
    VAE.to(device)

    X_training = dat['X_train'].to(device)
    fixed_noise = torch.randn(args.num_gen_images, args.nz, 1, 1, device=device)
    #optimizerE = optim.Adam(netE.parameters(), lr=args.lrD, betas=(args.beta1, 0.999))
    #optimizerG = optim.Adam(netG.parameters(), lr=args.lrG, betas=(args.beta1, 0.999)) 

## print generator.parameters to make sure G is fixed
    optimizerVAE = optim.Adam(VAE.parameters(), lr=args.lrE, betas=(args.beta1, 0.999))
    for epoch in range(1, args.epochs+1):
        for i in range(0, len(X_training), args.batchSize):
            stop = min(args.batchSize, len(X_training[i:]))
            real_cpu = X_training[i:i+stop].to(device)

            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device, dtype=torch.int8)

            zout, mu, log_var = VAE.forward(real_cpu)
            #pdb.set_trace()

            # sample z from q
            std = torch.exp(log_var / 2)
            q = torch.distributions.Normal(mu, std)
            z = q.rsample(sample_shape=torch.Size([]))  ## sample z from q
            z = torch.unsqueeze(z, 2)  
            z = torch.unsqueeze(z, 3)
            #pdb.set_trace()

            # decoded - GAN Generator
            outputG = netG(z)
            #pdb.set_trace()

            # reconstruction loss --> log_pxz
            log_scale = nn.Parameter(torch.Tensor([0.0]))
            logscale = log_scale.to(device)
            recon_loss = VAE.gaussian_likelihood(outputG, logscale, real_cpu)
            #pdb.set_trace()

            # kl ---> (log_qzx - log_pz)
            kl = VAE.kl_divergence(z, mu, std)
            #pdb.set_trace()

            # elbo --> -1 * (kl - recon_loss)
            #elbo = -1 * (kl - recon_loss)

            # elbo --> [(log_qzx_sum - log_pz_sum) - log_pxz_sum]
            log_qzx_sum, log_pz_sum, log_pxz_sum, ELBO = VAE.PGAN_ELBO(z, mu, std, outputG, logscale, real_cpu)
            elbo = (log_pxz_sum + log_pz_sum - log_qzx_sum)
            #pdb.set_trace()
            
            elbo = elbo.mean()
            #pdb.set_trace()

            #errE = criterion_mse(real_cpu, outputG,args)
            errE = elbo

            errE.backward()
            E_x = outputG.mean().item()
            optimizerVAE.step()

            ## log performance
            if i % args.log == 0:
                #print('Epoch [%d/%d] .. Batch [%d/%d] .. Loss_E: %.4f ..(kl: %.4f, recon_loss: %.4f) .. E(x): %.4f'
                #        % (epoch, args.epochs, i, len(X_training), errE.data, kl.mean(), recon_loss.mean(), E_x))

                print('Epoch [%d/%d] .. Batch [%d/%d] .. elbo: %.4f ..(log_pxz: %.4f, log_pz: %.4f, log_qzx: %.4f)'
                        % (epoch, args.epochs, i, len(X_training), elbo, log_pxz_sum.mean(), log_pz_sum.mean(), log_qzx_sum.mean()))

                #log performance to tensorboard
                writer.add_scalar("elbo", errE.data, epoch)
                #writer.add_scalar("E_x", E_x, epoch)
                writer.add_scalar("kl_divergence", kl.mean(), epoch)
                writer.add_scalar("recon_loss", recon_loss.mean(), epoch)
                writer.add_scalar("elbo_parts/log_pxz_sum", log_pxz_sum.mean(), epoch)
                writer.add_scalar("elbo_parts/log_pz_sum", log_pz_sum.mean(), epoch)
                writer.add_scalar("elbo_parts/log_qzx_sum", log_qzx_sum.mean(), epoch)
                writer.add_scalar("log_scale", log_scale, epoch)

                #-------------

        print('*'*100)
        print('End of epoch {}'.format(epoch))
        print('*'*100)

        if epoch % args.save_ckpt_every == 0:
            torch.save(netG.state_dict(), os.path.join(args.results_folder, 'netG_dcgan_%s_epoch_%s.pth'%(args.dataset, epoch)))

    writer.flush()
