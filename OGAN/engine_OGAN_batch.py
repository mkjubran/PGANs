import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import datetime
import pdb
import numpy as np
import math
import torchvision.utils as vutils

from scipy.stats import truncnorm
from scipy.stats import mvn as scipy_mvn
from scipy.stats import norm as scipy_norm

import TruncatedNormal as TNorm

##-- loading get distribution
def dist(args, device, mu, logvar, mean, scale, data, zr):
 imageSize = args.imageSize

 ##-- compute MVN full batch
 mvn = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(scale).reshape(1, imageSize*imageSize, imageSize*imageSize), validate_args=False)
 log_pxz_mvn = mvn.log_prob(data.view(-1,imageSize*imageSize))
 log_pxz_mvn = torch.sum(log_pxz_mvn.view(-1,args.nc),1)

 std = torch.exp(0.5*logvar)
 #std = torch.exp(logvar)
 std_b = torch.eye(std.size(1)).to(device)
 std_c = std.unsqueeze(2).expand(*std.size(), std.size(1))
 std_3d = std_c * std_b
 mvnz = torch.distributions.MultivariateNormal(mu, scale_tril=std_3d, validate_args = False)
 log_pz_normal = mvnz.log_prob(zr)
 return log_pxz_mvn, log_pz_normal

def get_likelihood(args, device, netE, optimizerE, data, netG, logsigmaG, ckptOL, logvar_first):
 #log_dir = ckptOL+"/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
 #writer = SummaryWriter(log_dir)

 netE.train()
 netG.eval()
 running_loss = 0.0
 counter = 0
 train_loss = []
 overlap_loss = 0;
 OLepoch = 0;
 scale = 0.1*torch.ones(args.imageSize**2).to(device)
 overlap_loss_sum = 1
 overlap_loss_min = args.overlap_loss_min
 while (OLepoch <= args.OLepochs) and (overlap_loss_sum > overlap_loss_min):
        OLepoch +=1
        counter += 1
        optimizerE.zero_grad()

        data=data.detach()

        #x_hat, mu, logvar, z, zr = netE(data, netG)
        mu, logvar, z, zr = netE(data, args)

        x_hat = netG(z)

        #if counter == 1:
          #logvar_first = logvar
          #logvar_first = 0.1*torch.ones(logvar.shape).to(device)

        mean = x_hat.view(-1,args.imageSize*args.imageSize)

        log_pxz_mvn, log_pz_normal = dist(args, device, mu, logvar, mean, scale, data, zr)

        ##-- definning overlap loss abd backpropagation
        overlap_loss = -1*(log_pxz_mvn + log_pz_normal) ## results of option#1
        overlap_loss_sum = overlap_loss.sum()

        if (overlap_loss_sum > overlap_loss_min):
           overlap_loss_sum.backward()
           running_loss += overlap_loss_sum.item()
           optimizerE.step()

        train_loss = running_loss / counter
      
        ##-- printing only the positive overlap loss (to avoid printing extremely low numbers after training coverage to low positive value)
        if (overlap_loss_sum > overlap_loss_min):
            likelihood_sample_final = overlap_loss_sum
            #writer.add_scalar("Train Loss/total", overlap_loss_sum, OLepoch)

        #-- write to tensorboard
        #if (OLepoch % 10 == 0) or (torch.isnan(z).unique()) or (OLepoch == 1):
        #    img_grid_TB = torchvision.utils.make_grid(torch.cat((data, x_hat), 3).detach().cpu(),nrow=3)
        #    writer.add_image('True (or sampled) images and Recon images', img_grid_TB, OLepoch)

        #if (OLepoch % 100 == 0) or (torch.isnan(z).unique()) or (OLepoch == 1):
        #    vutils.save_image(torch.cat((data, x_hat), 3).detach().cpu(), '%s/elbo_OLepoch_%03d.png' % (log_dir, OLepoch), normalize=True, nrow=10) 

 likelihood_final = torch.tensor([1]).float()
 likelihood_final[likelihood_final==1]=float("NaN")
 k=1.2
 #if (counter > 1):
 if True:

  ##-- Create a standard MVN
  mean = torch.zeros(mu.shape[0],args.nzg).to(device)
  scale = torch.ones(mu.shape[0],args.nzg).to(device)
  std = scale
  std_b = torch.eye(std.size(1)).to(device)
  std_c = std.unsqueeze(2).expand(*std.size(), std.size(1))
  std_3d = std_c * std_b
  mvns = torch.distributions.MultivariateNormal(mean, scale_tril=std_3d, validate_args = False)

  ##-- Create the proposal, i.e Multivariate Normal with mean = z and CovMatrix = k*torch.exp(0.5*logvar_first)
  mean = mu.view([-1,args.nzg]).to(device)
  #std = k*torch.exp(0.5*logvar_first)
  std = k*torch.ones(scale.shape).to(device) 
  std_b = torch.eye(std.size(1)).to(device)
  std_c = std.unsqueeze(2).expand(*std.size(), std.size(1))
  std_3d = std_c * std_b
  mvnz = torch.distributions.MultivariateNormal(mean, scale_tril=std_3d, validate_args = False)

  sample_shape = torch.Size([])

  likelihood_sample_final = 0
  log_likelihood_sample_list = torch.tensor([]).to(device)
  ## sample and compute
  S = args.S
  samples = mvnz.sample((S,))
  log_pz = mvns.log_prob(samples)
  log_rzx = mvnz.log_prob(samples)

  log_pxz_scipy = 0
  Ta = torch.tensor([-1.]).to(device)
  Tb = torch.tensor([1.]).to(device)

  ##---------------- step is the number of images to be considered in parallel
  if samples.shape[1] > 5:
     step=5
  else:
     step=samples.shape[1]
  scale = torch.exp(0.5*logsigmaG).to(device).detach()
  for cnt in range(0,samples.shape[1],step):
    sample = samples[:,cnt:cnt+step,:].reshape(S*step,args.nzg)
    meanG = netG(sample.view(-1,args.nzg,1,1)).view(-1,args.imageSize*args.imageSize).to(device).detach()
    x = data[cnt:cnt+step].view(-1, args.imageSize**2).repeat(S,1).to(device)

    ## Method #1): using Truncated Normal Class from Github https://github.com/toshas/torch_truncnorm
    ## can be used with S > 1
    pt = TNorm.TruncatedNormal(meanG, scale, Ta, Tb, validate_args=None)
    if cnt == 0:
       log_pxz_scipy = torch.sum(pt.log_prob(x).view(S*step,-1), axis=1).view(S,step)
    else:
       log_pxz_scipy = torch.cat((log_pxz_scipy,torch.sum(pt.log_prob(x).view(S*step,-1), axis=1).view(S,step)),1)
  log_likelihood_samples = (log_pxz_scipy + log_pz - log_rzx)
  likelihood_samples = torch.log(torch.tensor(1/S))+torch.logsumexp(log_likelihood_samples,0)
  likelihood_final = torch.mean(likelihood_samples,0)
  #likelihood_final = torch.logsumexp(likelihood_samples,0)-torch.log(torch.tensor(likelihood_samples.shape[0]))
  #print(torch.logsumexp(likelihood_samples,0)-torch.log(torch.tensor(likelihood_samples.shape[0])))
  ##---------------- very fast, however it Needs a lot of GPU memory, fail for large S.
  '''
  sample = samples.view(S*samples.shape[1],args.nzg)
  meanG = netG(sample.view(-1,args.nzg,1,1)).view(-1,args.imageSize*args.imageSize).to(device).detach()
  scale = torch.exp(0.5*logsigmaG).to(device).detach()
  x = data.view(-1, args.imageSize**2).repeat(S,1).to(device)

  ## Method #1): using Truncated Normal Class from Github https://github.com/toshas/torch_truncnorm
  ## can be used with S > 1
  pt = TNorm.TruncatedNormal(meanG, scale, Ta, Tb, validate_args=None)
  log_pxz_scipy = torch.sum(pt.log_prob(x), axis=1).view(S,samples.shape[1])

  #print(datetime.datetime.now()-now)
  log_likelihood_samples = (log_pxz_scipy + log_pz - log_rzx)
  likelihood_samples = torch.log(torch.tensor(1/S))+torch.logsumexp(log_likelihood_samples,0)
  likelihood_final = torch.logsumexp(likelihood_samples,0)-torch.log(torch.tensor(likelihood_samples.shape[0]))
  '''
  ##----------------

  #img_grid_TB = torchvision.utils.make_grid(torch.cat((data.view(args.imageSize,args.imageSize), meanG.view(args.imageSize,args.imageSize)), 0).detach().cpu())
  #writer.add_image('mean of z', img_grid_TB, iter)

 #likelihood_sample_final = torch.log(likelihood_sample_final/S)
 #writer.add_scalar("Likelihood/likelihood_sample_final", likelihood_sample_final, Counter)

 #writer.flush()
 #writer.close()

 return likelihood_final

def get_likelihood_approx(args, device, netE, optimizerE, data, netG, logsigmaG, ckptOL, logvar_first):
 #log_dir = ckptOL+"/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
 #writer = SummaryWriter(log_dir)

 netE.train()
 netG.eval()
 running_loss = 0.0
 counter = 0
 train_loss = []
 overlap_loss = 0;
 OLepoch = 0;
 scale = 0.1*torch.ones(args.imageSize**2).to(device)
 overlap_loss_sum = 1
 overlap_loss_min = args.overlap_loss_min
 while (OLepoch <= args.OLepochs) and (overlap_loss_sum > overlap_loss_min):
        OLepoch +=1
        counter += 1
        optimizerE.zero_grad()

        data=data.detach()

        #x_hat, mu, logvar, z, zr = netE(data, netG)
        mu, logvar, z, zr = netE(data, args)

        x_hat = netG(z)

        #if counter == 1:
          #pdb.set_trace()
          #logvar_first = logvar
          #logvar_first = 0.1*torch.ones(logvar.shape).to(device)

        mean = x_hat.view(-1,args.imageSize*args.imageSize)

        log_pxz_mvn, log_pz_normal = dist(args, device, mu, logvar, mean, scale, data, zr)

        ##-- definning overlap loss abd backpropagation
        overlap_loss = -1*(log_pxz_mvn + log_pz_normal) ## results of option#1
        overlap_loss_sum = overlap_loss.sum()

        if (overlap_loss_sum > overlap_loss_min):
           overlap_loss_sum.backward()
           running_loss += overlap_loss_sum.item()
           optimizerE.step()

        train_loss = running_loss / counter

        ##-- printing only the positive overlap loss (to avoid printing extremely low numbers after training coverage to low positive value)
        if (overlap_loss_sum > overlap_loss_min):
            likelihood_sample_final = overlap_loss_sum
        #    writer.add_scalar("Train Loss/total", overlap_loss_sum, OLepoch)

        #-- write to tensorboard
        #if (OLepoch % 10 == 0) or (torch.isnan(z).unique()) or (OLepoch == 1):
        #    img_grid_TB = torchvision.utils.make_grid(torch.cat((data, x_hat), 3).detach().cpu(),nrow=3)
        #    writer.add_image('True (or sampled) images and Recon images', img_grid_TB, OLepoch)

        #if (OLepoch % 100 == 0) or (torch.isnan(z).unique()) or (OLepoch == 1):
        #    vutils.save_image(torch.cat((data, x_hat), 3).detach().cpu(), '%s/elbo_OLepoch_%03d.png' % (log_dir, OLepoch), normalize=True, nrow=10) 


 likelihood_final = torch.tensor([1]).float()
 likelihood_final[likelihood_final==1]=float("NaN")
 k=1.2
 #if (counter > 1):
 if True:

  ##-- Create a standard MVN
  mean = torch.zeros(mu.shape[0],args.nzg).to(device)
  scale = torch.ones(mu.shape[0],args.nzg).to(device)
  std = scale
  std_b = torch.eye(std.size(1)).to(device)
  std_c = std.unsqueeze(2).expand(*std.size(), std.size(1))
  std_3d = std_c * std_b
  mvns = torch.distributions.MultivariateNormal(mean, scale_tril=std_3d, validate_args = False)

  ##-- Create the proposal, i.e Multivariate Normal with mean = z and CovMatrix = k*torch.exp(0.5*logvar_first)
  mean = mu.view([-1,args.nzg]).to(device)
  #std = k*torch.exp(0.5*logvar_first)
  std = k*torch.ones(scale.shape).to(device)  
  std_b = torch.eye(std.size(1)).to(device)
  std_c = std.unsqueeze(2).expand(*std.size(), std.size(1))
  std_3d = std_c * std_b
  mvnz = torch.distributions.MultivariateNormal(mean, scale_tril=std_3d, validate_args = False)

  sample_shape = torch.Size([])

  likelihood_sample_final = 0
  log_likelihood_sample_list = torch.tensor([]).to(device)
  ## sample and compute
  S = args.S
  samples = mvnz.sample((S,))
  log_pz = mvns.log_prob(samples)
  log_rzx = mvnz.log_prob(samples)

  log_pxz_scipy = 0
  Ta = torch.tensor([-1.]).to(device)
  Tb = torch.tensor([1.]).to(device)

  std =  torch.diag(torch.exp(0.5*logsigmaG)).reshape(1, args.imageSize*args.imageSize, args.imageSize*args.imageSize).to(device).detach()
  for cnt in range(samples.shape[1]):
    sample = samples[:,cnt,:] #.detach()
    meanG = netG(sample.view(-1,args.nzg,1,1)).view(-1,args.imageSize*args.imageSize).to(device).detach()
    x = data[cnt].view(args.nc, args.imageSize**2).to(device)
    x = x.repeat(S,1)
    mvnx = torch.distributions.MultivariateNormal(meanG, scale_tril=std, validate_args=False)
    if cnt == 0:
       log_pxz_scipy = torch.sum(mvnx.log_prob(x).view(S,args.nc),1).view(-1,1)
    else:
       log_pxz_scipy = torch.cat((log_pxz_scipy,torch.sum(mvnx.log_prob(x).view(S,args.nc),1).view(-1,1)),1)
  log_likelihood_samples = (log_pxz_scipy + log_pz - log_rzx)
  likelihood_samples = torch.log(torch.tensor(1/S))+torch.logsumexp(log_likelihood_samples,0)
  likelihood_final = torch.mean(likelihood_samples)
  #likelihood_final = torch.logsumexp(likelihood_samples,0)-torch.log(torch.tensor(likelihood_samples.shape[0]))

 #writer.flush()
 #writer.close()

 return likelihood_final


def get_likelihood_VAE(args, device, netE, optimizerE, data, netDec, ckptOL):
 #log_dir = ckptOL+"/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
 #writer = SummaryWriter(log_dir)

 netE.train()
 running_loss = 0.0
 counter = 0
 train_loss = []
 overlap_loss = 0;
 OLepoch = 0;
 scale = 0.1*torch.ones(args.imageSize**2).to(device)
 overlap_loss_sum = 1
 overlap_loss_min = args.overlap_loss_min
 while (OLepoch <= args.OLepochs) and (overlap_loss_sum > overlap_loss_min):
        OLepoch +=1
        counter += 1
        optimizerE.zero_grad()
        data=data.detach()

        #x_hat, mu, logvar, z, zr = netE(data, netG)
        mu, logvar, z, zr = netE(data, args)

        x_hat = netDec(z,args)

        #if counter == 1:
        #  logvar_first = logvar

        mean = x_hat.view(-1,args.imageSize*args.imageSize)

        log_pxz_mvn, log_pz_normal = dist(args, device, mu, logvar, mean, scale, data, zr)

        ##-- definning overlap loss abd backpropagation
        overlap_loss = -1*(log_pxz_mvn + log_pz_normal) ## results of option#1
        overlap_loss_sum = overlap_loss.sum()

        if (overlap_loss_sum > overlap_loss_min):
           overlap_loss_sum.backward()
           running_loss += overlap_loss_sum.item()
           optimizerE.step()

        train_loss = running_loss / counter

        ##-- printing only the positive overlap loss (to avoid printing extremely low numbers after training coverage to low positive value)
        if overlap_loss_sum > overlap_loss_min:
            likelihood_sample_final = overlap_loss_sum
        #    writer.add_scalar("Train Loss/total", overlap_loss_sum, OLepoch)

        #-- write to tensorboard
        #if (OLepoch % 50 == 0) or (torch.isnan(z).unique()) or (OLepoch == 1):
        #    img_grid_TB = torchvision.utils.make_grid(torch.cat((data, x_hat), 3).detach().cpu(),nrow=3)
        #    writer.add_image('True (or sampled) images and Recon images', img_grid_TB, OLepoch)


 likelihood_final = torch.tensor([1]).float()
 likelihood_final[likelihood_final==1]=float("NaN")
 k = 1.2
 if True:
  ##-- Create a standard MVN
  mean = torch.zeros(mu.shape[0],args.nzg).to(device)
  scale = torch.ones(mu.shape[0],args.nzg).to(device)
  std = scale
  std_b = torch.eye(std.size(1)).to(device)
  std_c = std.unsqueeze(2).expand(*std.size(), std.size(1))
  std_3d = std_c * std_b
  mvns = torch.distributions.MultivariateNormal(mean, scale_tril=std_3d, validate_args = False)

  ##-- Create the proposal, i.e Multivariate Normal with mean = z and CovMatrix = 0.01
  mean = mu.view([-1,args.nzg]).to(device)
  #std = k*torch.exp(0.5*logvar_first)
  std = k*torch.ones(scale.shape).to(device)  ## no logsigma for VAE
  std_b = torch.eye(std.size(1)).to(device)
  std_c = std.unsqueeze(2).expand(*std.size(), std.size(1))
  std_3d = std_c * std_b
  mvnz = torch.distributions.MultivariateNormal(mean, scale_tril=std_3d, validate_args = False)
  sample_shape = torch.Size([])

  likelihood_sample_final = 0
  log_likelihood_sample_list = torch.tensor([]).to(device)
  ## sample and compute
  S = args.S
  samples = mvnz.sample((S,))
  log_pz = mvns.log_prob(samples)
  log_rzx = mvnz.log_prob(samples)

  log_pxz_scipy = 0
  Ta = torch.tensor([-1.]).to(device)
  Tb = torch.tensor([1.]).to(device)

 ##---------------- step is the number of images to be considered in parallel
  step=5
  #scale = torch.exp(0.5*logsigmaG).to(device).detach()
  scale = k*torch.ones(args.imageSize**2).to(device).detach()  ## no logsigma for VAE
  for cnt in range(0,samples.shape[1],step):
    sample = samples[:,cnt:cnt+step,:].reshape(S*step,args.nzg)
    meanG = netDec(sample.view(-1,args.nzg,1,1), args).view(-1,args.imageSize*args.imageSize).to(device).detach()
    x = data[cnt:cnt+step].view(-1, args.imageSize**2).repeat(S,1).to(device)

    pt = TNorm.TruncatedNormal(meanG, scale, Ta, Tb, validate_args=None)
    if cnt == 0:
       log_pxz_scipy = torch.sum(pt.log_prob(x), axis=1).view(S,step)
    else:
       log_pxz_scipy = torch.cat((log_pxz_scipy,torch.sum(pt.log_prob(x), axis=1).view(S,step)),1)
  log_likelihood_samples = (log_pxz_scipy + log_pz - log_rzx)
  likelihood_samples = torch.log(torch.tensor(1/S))+torch.logsumexp(log_likelihood_samples,0)
  likelihood_final = torch.mean(likelihood_samples,0)
  #likelihood_final = torch.logsumexp(likelihood_samples,0)-torch.log(torch.tensor(likelihood_samples.shape[0]))

 return likelihood_final


def get_likelihood_MLL(args, device, netE, optimizerE, data, netG, logsigmaG, ckptOL, logvar_first):
 #log_dir = ckptOL+"/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
 #writer = SummaryWriter(log_dir)

 netE.train()
 netG.eval()
 running_loss = 0.0
 counter = 0
 train_loss = []
 overlap_loss = 0;
 OLepoch = 0;
 scale = 0.1*torch.ones(args.imageSize**2).to(device)
 overlap_loss_sum = 1
 overlap_loss_min = args.overlap_loss_min
 while (OLepoch <= args.OLepochs) and (overlap_loss_sum > overlap_loss_min):
        OLepoch +=1
        counter += 1
        optimizerE.zero_grad()

        data=data.detach()

        #x_hat, mu, logvar, z, zr = netE(data, netG)
        mu, logvar, z, zr = netE(data, args)

        x_hat = netG(z)

        #if counter == 1:
          #logvar_first = logvar
          #logvar_first = 0.1*torch.ones(logvar.shape).to(device)

        mean = x_hat.view(-1,args.imageSize*args.imageSize)

        log_pxz_mvn, log_pz_normal = dist(args, device, mu, logvar, mean, scale, data, zr)

        ##-- definning overlap loss abd backpropagation
        overlap_loss = -1*(log_pxz_mvn + log_pz_normal) ## results of option#1
        overlap_loss_sum = overlap_loss.sum()

        if (overlap_loss_sum > overlap_loss_min):
           overlap_loss_sum.backward()
           running_loss += overlap_loss_sum.item()
           optimizerE.step()

        train_loss = running_loss / counter
      
        ##-- printing only the positive overlap loss (to avoid printing extremely low numbers after training coverage to low positive value)
        if (overlap_loss_sum > overlap_loss_min):
            likelihood_sample_final = overlap_loss_sum
            #writer.add_scalar("Train Loss/total", overlap_loss_sum, OLepoch)

        #-- write to tensorboard
        #if (OLepoch % 10 == 0) or (torch.isnan(z).unique()) or (OLepoch == 1):
        #    img_grid_TB = torchvision.utils.make_grid(torch.cat((data, x_hat), 3).detach().cpu(),nrow=3)
        #    writer.add_image('True (or sampled) images and Recon images', img_grid_TB, OLepoch)

        #if (OLepoch % 100 == 0) or (torch.isnan(z).unique()) or (OLepoch == 1):
        #    vutils.save_image(torch.cat((data, x_hat), 3).detach().cpu(), '%s/elbo_OLepoch_%03d.png' % (log_dir, OLepoch), normalize=True, nrow=10) 

 likelihood_final = torch.tensor([1]).float()
 likelihood_final[likelihood_final==1]=float("NaN")
 k=1.2
 #if (counter > 1):
 if True:

  ##-- Create a standard MVN
  mean = torch.zeros(mu.shape[0],args.nzg).to(device)
  scale = torch.ones(mu.shape[0],args.nzg).to(device)
  std = scale
  std_b = torch.eye(std.size(1)).to(device)
  std_c = std.unsqueeze(2).expand(*std.size(), std.size(1))
  std_3d = std_c * std_b
  mvns = torch.distributions.MultivariateNormal(mean, scale_tril=std_3d, validate_args = False)

  ##-- Create the proposal, i.e Multivariate Normal with mean = z and CovMatrix = k*torch.exp(0.5*logvar_first)
  mean = mu.view([-1,args.nzg]).to(device)
  #std = k*torch.exp(0.5*logvar_first)
  std = k*torch.ones(scale.shape).to(device) 
  std_b = torch.eye(std.size(1)).to(device)
  std_c = std.unsqueeze(2).expand(*std.size(), std.size(1))
  std_3d = std_c * std_b
  mvnz = torch.distributions.MultivariateNormal(mean, scale_tril=std_3d, validate_args = False)

  sample_shape = torch.Size([])

  likelihood_sample_final = 0
  log_likelihood_sample_list = torch.tensor([]).to(device)
  ## sample and compute
  #S = args.S
  for cntS in range(0,args.S,1000):
    S=1000
    samples = mvnz.sample((S,))
    log_pz = mvns.log_prob(samples)
    log_rzx = mvnz.log_prob(samples)

    log_pxz_scipy = 0
    Ta = torch.tensor([-1.]).to(device)
    Tb = torch.tensor([1.]).to(device)

    ##---------------- step is the number of images to be considered in parallel
    if samples.shape[1] > 5:
       step=5
    else:
       step=samples.shape[1]
    scale = torch.exp(0.5*logsigmaG).to(device).detach()
    for cnt in range(0,samples.shape[1],step):
      sample = samples[:,cnt:cnt+step,:].reshape(S*step,args.nzg)
      meanG = netG(sample.view(-1,args.nzg,1,1)).view(-1,args.imageSize*args.imageSize).to(device).detach()
      x = data[cnt:cnt+step].view(-1, args.imageSize**2).repeat(S,1).to(device)

      ## Method #1): using Truncated Normal Class from Github https://github.com/toshas/torch_truncnorm
      ## can be used with S > 1
      pt = TNorm.TruncatedNormal(meanG, scale, Ta, Tb, validate_args=None)
      if cnt == 0:
         log_pxz_scipy = torch.sum(pt.log_prob(x).view(S*step,-1), axis=1).view(S,step)
      else:
         log_pxz_scipy = torch.cat((log_pxz_scipy,torch.sum(pt.log_prob(x).view(S*step,-1), axis=1).view(S,step)),1)

    if cntS == 0:
       log_pz_argsS = log_pz
       log_rzx_argsS = log_rzx
       log_pxz_scipy_argsS = log_pxz_scipy
    else:
       log_pz_argsS = torch.cat((log_pz_argsS,log_pz),0)
       log_rzx_argsS = torch.cat((log_rzx_argsS,log_rzx),0)
       log_pxz_scipy_argsS = torch.cat((log_pxz_scipy_argsS,log_pxz_scipy),0)

  log_likelihood_samples = (log_pxz_scipy_argsS + log_pz_argsS - log_rzx_argsS)
  likelihood_samples = torch.log(torch.tensor(1/args.S))+torch.logsumexp(log_likelihood_samples,0)
  likelihood_final = torch.mean(likelihood_samples,0)

 #writer.flush()
 #writer.close()

 return likelihood_samples
