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
import statistics

import utils
import hmc
import pdb

from torch.distributions.normal import Normal

real_label = 1
fake_label = 0
criterion = nn.BCELoss()
criterion_mse = nn.MSELoss()


def dcgan(args, device, dat, netG, netD):
    real_label = 1
    fake_label = 0
    criterion = nn.BCELoss()
    criterion_mse = nn.MSELoss()

    writer = SummaryWriter(args.results_folder)
    device = args.device
    X_training = dat['X_train'].to(device)
    fixed_noise = torch.randn(args.num_gen_images, args.nz, 1, 1, device=device)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lrD, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lrG, betas=(args.beta1, 0.999))
    for epoch in range(1, args.epochs+1):
        DL=0
        GL=0
        Dx=0
        DL_G_z1=0
        DL_G_z2=0
        Counter=0
        for i in range(0, len(X_training), args.batchSize):
            Counter=Counter+1
            netD.zero_grad()
            stop = min(args.batchSize, len(X_training[i:]))
            real_cpu = X_training[i:i+stop].to(device)

            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device, dtype=torch.int8)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = torch.randn(batch_size, args.nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            # (2) Update G network: maximize log(D(G(z)))

            netG.zero_grad()
            label.fill_(real_label) 
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            ## log performance
            if i % args.log == 0:
                print('Epoch [%d/%d] .. Batch [%d/%d] .. Loss_D: %.4f .. Loss_G: %.4f .. D(x): %.4f .. D(G(z)): %.4f / %.4f'
                        % (epoch, args.epochs, i, int(len(X_training)/args.batchSize), errD.data, errG.data, D_x, D_G_z1, D_G_z2))

            DL = DL + errD.data
            GL = GL + g_error_gan.data
            Dx = Dx + D_x
            DL_G_z1 = DL_G_z1 + D_G_z1
            DL_G_z2 = DL_G_z2 + D_G_z2

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
        #-------------
        #pdb.set_trace()

        print('*'*100)
        print('End of epoch {}'.format(epoch))
        print('Epoch [%d/%d] .. Loss_D: %.4f .. Loss_G: %.4f .. D(x): %.4f .. D(G(z)): %.4f / %.4f'
                        % (epoch, args.epochs, DL, GL, Dx, DL_G_z1, DL_G_z2))

        print('*'*100)

       
        if epoch % args.save_imgs_every == 0:
            fake = netG(fixed_noise).detach()
            vutils.save_image(fake, '%s/dcgan_%s_fake_epoch_%03d.png' % (args.results_folder, args.dataset, epoch), normalize=True, nrow=20) 


            # log images to tensorboard
            # create grid of images
            img_grid = torchvision.utils.make_grid(fake)

            # write to tensorboard
            writer.add_image('fake_images', img_grid, epoch)
            # --------------


        if epoch % args.save_ckpt_every == 0:
            torch.save(netG.state_dict(), os.path.join(args.results_folder, 'netG_dcgan_%s_epoch_%s.pth'%(args.dataset, epoch)))
            torch.save(netD.state_dict(), os.path.join(args.results_folder, 'netD_dcgan_%s_epoch_%s.pth'%(args.dataset, epoch)))

    writer.flush()


def presgan(args, device, epoch, dat, netG, optimizerG, netD, optimizerD, log_sigma, sigma_optimizer, OLoss, ckptOLG, save_imgs,generator, Counter_epoch_batch):
    real_label = 1
    fake_label = 0
    criterion = nn.BCELoss()
    criterion_mse = nn.MSELoss()

    writer = SummaryWriter(ckptOLG)
    X_training = dat.to(device)
    fixed_noise = torch.randn(args.num_gen_images, args.nzg, 1, 1, device=device)
    if args.restrict_sigma:
        logsigma_min = math.log(math.exp(args.sigma_min) - 1.0)
        logsigma_max = math.log(math.exp(args.sigma_max) - 1.0)
    stepsize = args.stepsize_num / args.nzg

    bsz = args.batchSize
    i = 0
            
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
    #pdb.set_trace()
    g_error_gan = g_error_gan + OLoss
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
    #if i % args.log == 0:
    #   print('Epoch [%d/%d] .. Batch [%d/%d] .. Loss_D: %.4f .. Loss_G: %.4f .. D(x): %.4f .. D(G(z)): %.4f / %.4f'
    #       % (epoch, args.epochs, Counter, len(X_training), errD.data, g_error_gan.data, D_x, D_G_z1, D_G_z2))

    DL = errD.data
    GL = g_error_gan.data
    Dx = D_x
    DL_G_z1 = D_G_z1
    DL_G_z2 = D_G_z2

    PresGANResults=[DL, GL, Dx, DL_G_z1, DL_G_z2, torch.min(sigma_x), torch.max(sigma_x)]

    if save_imgs:
        fake = netG(fixed_noise).detach()
    #    vutils.save_image(fake, '%s/presgan_%s_fake_epoch_%03d.png' % (args.results_folder, args.dataset, epoch), normalize=True, nrow=20)

         # log images to tensorboard
         # create grid of images
        img_grid = torchvision.utils.make_grid(fake)

         # write to tensorboard
        if generator == 'G1':
            writer.add_image('G1-fake_images', img_grid, Counter_epoch_batch)
        else:
            writer.add_image('G2-fake_images', img_grid, Counter_epoch_batch)
         # --------------

    writer.flush()
    return netD, netG, log_sigma, PresGANResults

