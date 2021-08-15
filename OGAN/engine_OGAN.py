import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import datetime
import pdb
import numpy as np
import math

from scipy.stats import truncnorm
from scipy.stats import mvn as scipy_mvn
from scipy.stats import norm as scipy_norm

import TruncatedNormal as TNorm

##-- loading get distribution
def dist(args, device, mu, logvar, mean, scale, data, zr):
 imageSize = args.imageSize

 ##-- compute MVN full batch
 mvn = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(scale).reshape(1, imageSize*imageSize, imageSize*imageSize))
 log_pxz_mvn = mvn.log_prob(data.view(-1,imageSize*imageSize))

 std = torch.exp(0.5*logvar)
 #std = torch.exp(logvar)
 std_b = torch.eye(std.size(1)).to(device)
 std_c = std.unsqueeze(2).expand(*std.size(), std.size(1))
 std_3d = std_c * std_b
 mvnz = torch.distributions.MultivariateNormal(mu, scale_tril=std_3d)
 log_pz_normal = mvnz.log_prob(zr)

 return log_pxz_mvn, log_pz_normal

def get_overlap_loss(args,device,netE,optimizerE,data,netG,scale,ckptOL):
 #log_dir = ckptOL+"/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
 #writer = SummaryWriter(log_dir)

 running_loss = 0.0
 counter = 0
 train_loss = []
 overlap_loss = 0;
 OLepoch = 0;
 while (OLepoch <= args.OLepochs) and (overlap_loss >= 0):
        OLepoch +=1
        counter += 1
        optimizerE.zero_grad()

        #x_hat, mu, logvar, z, zr = netE(data, netG)
        mu, logvar, z, zr = netE(data, args)
        x_hat = netG(z)

        mean = x_hat.view(-1,args.imageSize*args.imageSize)

        log_pxz_mvn, log_pz_normal = dist(args, device, mu, logvar, mean, scale, data, zr)

        ##-- definning overlap loss abd backpropagation 
        overlap_loss = -1*(log_pxz_mvn + log_pz_normal) ## results of option#1
        #overlap_loss = (torch.exp(log_pxz_mvn)) ## results of option#2 are not ready because torch.exp(log_pxz_mvn) always zero
        #pdb.set_trace()
        overlap_loss.backward()
        running_loss += overlap_loss.item()
        optimizerE.step()
        train_loss = running_loss / counter
        
        ##-- print training loss
        #if OLepoch % 5 ==0:
        #   print(f"Train Loss at epoch {epoch}: {train_loss:.4f}")

        ##-- printing only the positive overlap loss (to avoid printing extremely low numbers after training coverage to low positive value)
        if overlap_loss >= 0:
            overlap_loss_sample_final = overlap_loss
 #           writer.add_scalar("Train Loss", overlap_loss, OLepoch)

        ##-- write to tensorboard
 #       if OLepoch % 10 == 0:
 #           img_grid_TB = torchvision.utils.make_grid(torch.cat((data, x_hat), 0).detach().cpu())
 #           writer.add_image('True (or sampled) image and recon_image', img_grid_TB, OLepoch)
 #writer.flush()
 #writer.close()
 return overlap_loss_sample_final


def get_likelihood(args, device, netE, optimizerE, data, netG, logsigmaG, ckptOL):
 #log_dir = ckptOL+"/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
 #writer = SummaryWriter(log_dir)

 running_loss = 0.0
 counter = 0
 train_loss = []
 overlap_loss = 0;
 OLepoch = 0;
 scale = 0.1*torch.ones(args.imageSize**2).to(device)
 while (OLepoch <= args.OLepochs) and (overlap_loss >= 0):
        OLepoch +=1
        counter += 1
        optimizerE.zero_grad()

        #x_hat, mu, logvar, z, zr = netE(data, netG)
        mu, logvar, z, zr = netE(data, args)
        x_hat = netG(z)

        if counter == 1:
          logvar_first = logvar
        mean = x_hat.view(-1,args.imageSize*args.imageSize)

        log_pxz_mvn, log_pz_normal = dist(args, device, mu, logvar, mean, scale, data, zr)

        ##-- definning overlap loss abd backpropagation 
        overlap_loss = -1*(log_pxz_mvn + log_pz_normal) ## results of option#1
        #overlap_loss = (torch.exp(log_pxz_mvn)) ## results of option#2 are not ready because torch.exp(log_pxz_mvn) always zero
        #pdb.set_trace()
        overlap_loss.backward()
        running_loss += overlap_loss.item()
        optimizerE.step()
        train_loss = running_loss / counter
       
        ##-- print training loss
        #if OLepoch % 5 ==0:
        #   print(f"Train Loss at epoch {epoch}: {train_loss:.4f}")

        ##-- printing only the positive overlap loss (to avoid printing extremely low numbers after training coverage to low positive value)
        if overlap_loss >= 0:
            likelihood_sample_final = overlap_loss
            #writer.add_scalar("Train Loss/total", overlap_loss, OLepoch)
            #writer.add_scalar("Train Loss/log_pz_normal", log_pz_normal, OLepoch)
            #writer.add_scalar("Train Loss/log_pxz_mvn", log_pxz_mvn, OLepoch)

        ##-- write to tensorboard
        #if OLepoch % 10 == 0:
        #    img_grid_TB = torchvision.utils.make_grid(torch.cat((data, x_hat), 0).detach().cpu())
        #    writer.add_image('True (or sampled) image and recon_image', img_grid_TB, OLepoch)
 Counter = 0
 #for k in np.arange(0.001,0.1,0.0001):
 k = 0.1
 if True:
  Counter += 1
  ##-- Create a standard MVN
  mean = torch.zeros(args.nzg).to(device)
  scale = torch.ones(args.nzg).to(device)
  mvns = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(scale).view(1, args.nzg, args.nzg))

  ##-- Create the proposal, i.e Multivariate Normal with mean = z and CovMatrix = 0.01
  #mean = z.view([-1,args.nzg]).to(device)
  mean = mu.view([-1,args.nzg]).to(device)
  scale = k*torch.exp(0.5*logvar_first.view([args.nzg])).to(device)
  #scale = k*torch.ones(args.nzg).to(device)
  mvnz = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(scale).view(1, args.nzg, args.nzg))
  sample_shape = torch.Size([])

  likelihood_sample_final = 0
  log_likelihood_sample_list = torch.tensor([]).to(device)
  ## sample and compute
  S = args.S
  sample = mvnz.sample((S,))
  log_pz = mvns.log_prob(sample)
  log_rzx = mvnz.log_prob(sample)

  log_pxz_scipy = 0
  Ta = torch.tensor([-1.]).to(device)
  Tb = torch.tensor([1.]).to(device)
  meanG = netG(sample.view(-1,args.nzg,1,1)).view(-1,args.imageSize*args.imageSize).to(device)
  scale = torch.exp(0.5*logsigmaG).to(device)
  x = data.view(args.imageSize**2).to(device)

  ## Method #1): using Truncated Normal Class from Github https://github.com/toshas/torch_truncnorm
  ## can be used with S > 1
  pt = TNorm.TruncatedNormal(meanG, scale, Ta, Tb, validate_args=None)
  log_pxz_scipy = (torch.sum(pt.log_prob(x), axis=1)).view(log_rzx.shape)
  log_likelihood_sample = (log_pxz_scipy + log_pz - log_rzx)
  likelihood_sample = torch.log(torch.tensor(1/S))+torch.logsumexp(log_likelihood_sample,0)
  #print(likelihood_sample)

  '''
  ## Method #2: use Normal Distribution, ignore truncation as variance is very low
  ## can be used with S > 1
  pt = torch.distributions.MultivariateNormal(meanG, scale_tril=torch.diag(scale).view(1, args.imageSize*args.imageSize, args.imageSize*args.imageSize))
  log_pxz_scipy = pt.log_prob(x).view(log_pz.shape)
  log_likelihood_sample = (log_pxz_scipy + log_pz - log_rzx)
  likelihood_sample = torch.log(torch.tensor(1/S))+torch.logsumexp(log_likelihood_sample,0)
  print(likelihood_sample)
  '''

  '''
  ## Method #3: use scipy normal to get the truncated normal
  ## can be used only with S = 1
  if S == 1:
   log_pxz_scipy = 0
   mean = meanG.view(args.imageSize**2).detach().to('cpu')
   scale = scale.view(args.imageSize**2).detach().to('cpu')
   x = data.view(args.imageSize**2).detach().to('cpu')

   for cnt in range(0, args.imageSize**2):
     P = scipy_norm.pdf(x[cnt],mean[cnt],scale[cnt])
     P = max(P,0.0000000001)
     CDF_minus1=scipy_norm.cdf(-1,mean[cnt],scale[cnt])
     CDF_1=scipy_norm.cdf(1,mean[cnt],scale[cnt])
     log_pxz_scipy = log_pxz_scipy + np.log(P/(scale[cnt]*(CDF_1-CDF_minus1)))
   log_likelihood_sample = (log_pxz_scipy + log_pz - log_rzx)
   likelihood_sample = log_likelihood_sample
   print(likelihood_sample)
  '''

  '''
  ## Method 4: using the truncnorm of scipy
  log_likelihood_sample_list=[]
  scale_sci = scale.view(args.imageSize**2).detach().to('cpu')
  x_sci = data.view(args.imageSize**2).detach().to('cpu')
  meanG_sci = meanG.detach().to('cpu')
  for cnt in range(0,S):
    mean_sci = meanG_sci[cnt].view(args.imageSize**2)
    pxz_scipy = truncnorm.pdf(x_sci, a=-1, b=1, loc = mean_sci, scale = scale_sci)
    log_likelihood_sample = np.log(pxz_scipy) + log_pz.detach().cpu().numpy() - log_rzx.detach().cpu().numpy()
    log_likelihood_sample_list.append(np.sum(log_likelihood_sample))
    print(log_likelihood_sample_list)
  likelihood_sample = torch.log(torch.tensor(1/S))+torch.logsumexp(log_likelihood_sample,0)
  print(likelihood_sample)
  '''

  #img_grid_TB = torchvision.utils.make_grid(torch.cat((data.view(args.imageSize,args.imageSize), meanG.view(args.imageSize,args.imageSize)), 0).detach().cpu())
  #writer.add_image('mean of z', img_grid_TB, iter)

  #writer.add_scalar("Likelihood/log_pz", log_pz, Counter)
  #writer.add_scalar("Likelihood/log_rzx", log_rzx, Counter)
  #writer.add_scalar("Likelihood/log_pxz", log_pxz_scipy, Counter)
  #writer.add_scalar("Lihood/k - overdispersing hyperparameter", k, Counter)
  #writer.add_scalar("Likelihood/log_pxz+log_pz-log_rzx", log_likelihood_sample, Counter)
  #writer.add_scalar("Likelihood/(pxz*pz/rzx)", torch.exp(log_likelihood_sample), Counter)
  #pdb.set_trace()

 #likelihood_sample_final = torch.log(likelihood_sample_final/S)
 #writer.add_scalar("Likelihood/likelihood_sample_final", likelihood_sample_final, Counter)

 #writer.flush()
 #writer.close()

 return likelihood_sample

def get_likelihood_VAE(args, device, netE, optimizerE, data, netDec, ckptOL):
 #log_dir = ckptOL+"/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
 #writer = SummaryWriter(log_dir)

 running_loss = 0.0
 counter = 0
 train_loss = []
 overlap_loss = 0;
 OLepoch = 0;
 scale = 0.1*torch.ones(args.imageSize**2).to(device)
 while (OLepoch <= args.OLepochs) and (overlap_loss >= 0):
        OLepoch +=1
        counter += 1
        optimizerE.zero_grad()

        #x_hat, mu, logvar, z, zr = netE(data, netG)
        mu, logvar, z, zr = netE(data, args)
        x_hat = netDec(z,args)

        if counter == 1:
          logvar_first = logvar
        mean = x_hat.view(-1,args.imageSize*args.imageSize)

        log_pxz_mvn, log_pz_normal = dist(args, device, mu, logvar, mean, scale, data, zr)

        ##-- definning overlap loss abd backpropagation 
        overlap_loss = -1*(log_pxz_mvn + log_pz_normal) ## results of option#1
        #overlap_loss = (torch.exp(log_pxz_mvn)) ## results of option#2 are not ready because torch.exp(log_pxz_mvn) always zero
        #pdb.set_trace()
        overlap_loss.backward()
        running_loss += overlap_loss.item()
        optimizerE.step()
        train_loss = running_loss / counter
       
        ##-- print training loss
        #if OLepoch % 5 ==0:
        #   print(f"Train Loss at epoch {epoch}: {train_loss:.4f}")

        ##-- printing only the positive overlap loss (to avoid printing extremely low numbers after training coverage to low positive value)
        if overlap_loss >= 0:
            likelihood_sample_final = overlap_loss
            #writer.add_scalar("Train Loss/total", overlap_loss, OLepoch)
            #writer.add_scalar("Train Loss/log_pz_normal", log_pz_normal, OLepoch)
            #writer.add_scalar("Train Loss/log_pxz_mvn", log_pxz_mvn, OLepoch)

        ##-- write to tensorboard
        #if OLepoch % 10 == 0:
        #    img_grid_TB = torchvision.utils.make_grid(torch.cat((data, x_hat), 0).detach().cpu())
        #    writer.add_image('True (or sampled) image and recon_image', img_grid_TB, OLepoch)
 Counter = 0
 #for k in np.arange(0.001,0.1,0.0001):
 k = 0.1
 if True:
  Counter += 1
  ##-- Create a standard MVN
  mean = torch.zeros(args.nzg).to(device)
  scale = torch.ones(args.nzg).to(device)
  mvns = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(scale).view(1, args.nzg, args.nzg))

  ##-- Create the proposal, i.e Multivariate Normal with mean = z and CovMatrix = 0.01
  mean = mu.view([-1,args.nzg]).to(device)
  scale = k*torch.exp(0.5*logvar_first.view([args.nzg])).to(device)
  mvnz = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(scale).view(1, args.nzg, args.nzg))
  sample_shape = torch.Size([])

  likelihood_sample_final = 0
  log_likelihood_sample_list = torch.tensor([]).to(device)
  ## sample and compute
  S = args.S
  sample = mvnz.sample((S,))
  log_pz = mvns.log_prob(sample)
  log_rzx = mvnz.log_prob(sample)

  log_pxz_scipy = 0
  Ta = torch.tensor([-1.]).to(device)
  Tb = torch.tensor([1.]).to(device)
  meanG = netDec(sample.view(-1,args.nzg,1,1),args).view(-1,args.imageSize*args.imageSize).to(device)
  #scale = torch.exp(0.5*logsigmaG).to(device)  ## only for PresGAN
  scale = 10*k*torch.ones(args.imageSize**2).to(device)  ## no logsigma for VAE
  x = data.view(args.imageSize**2).to(device)

  ## Method #1): using Truncated Normal Class from Github https://github.com/toshas/torch_truncnorm
  ## can be used with S > 1
  pt = TNorm.TruncatedNormal(meanG, scale, Ta, Tb, validate_args=None)
  log_pxz_scipy = (torch.sum(pt.log_prob(x), axis=1)).view(log_rzx.shape)
  log_likelihood_sample = (log_pxz_scipy + log_pz - log_rzx)
  likelihood_sample = torch.log(torch.tensor(1/S))+torch.logsumexp(log_likelihood_sample,0)

  #img_grid_TB = torchvision.utils.make_grid(torch.cat((data.view(args.imageSize,args.imageSize), meanG.view(args.imageSize,args.imageSize)), 0).detach().cpu())
  #writer.add_image('mean of z', img_grid_TB, iter)

  #writer.add_scalar("Likelihood/log_pz", log_pz, Counter)
  #writer.add_scalar("Likelihood/log_rzx", log_rzx, Counter)
  #writer.add_scalar("Likelihood/log_pxz", log_pxz_scipy, Counter)
  #writer.add_scalar("Lihood/k - overdispersing hyperparameter", k, Counter)
  #writer.add_scalar("Likelihood/log_pxz+log_pz-log_rzx", log_likelihood_sample, Counter)
  #writer.add_scalar("Likelihood/(pxz*pz/rzx)", torch.exp(log_likelihood_sample), Counter)
  #pdb.set_trace()

 #likelihood_sample_final = torch.log(likelihood_sample_final/S)
 #writer.add_scalar("Likelihood/likelihood_sample_final", likelihood_sample_final, Counter)

 #writer.flush()
 #writer.close()

 return likelihood_sample
