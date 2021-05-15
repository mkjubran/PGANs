import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import datetime
import pdb
import numpy as np

##-- loading get distribution
def dist(args, device, mu, logvar, mean, scale, data, zr):
 imageSize = args.imageSize

 ##-- compute MVN full batch
 mvn = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(scale).reshape(1, imageSize*imageSize, imageSize*imageSize))
 log_pxz_mvn = mvn.log_prob(data.view(-1,imageSize*imageSize))

 ##-- sample from standard normal distribution
 std = torch.exp(0.5*logvar)
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
        x_hat, mu, logvar, z, zr = netE(data, netG)
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
 log_dir = ckptOL+"/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
 writer = SummaryWriter(log_dir)

 running_loss = 0.0
 counter = 0
 train_loss = []
 overlap_loss = 0;
 OLepoch = 0;
 scale = 0.01*torch.ones(args.imageSize**2).to(device)
 while (OLepoch <= args.OLepochs) and (overlap_loss >= 0):
        OLepoch +=1
        counter += 1
        optimizerE.zero_grad()
        x_hat, mu, logvar, z, zr = netE(data, netG)
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
            writer.add_scalar("Train Loss", overlap_loss, OLepoch)

        ##-- write to tensorboard
        if OLepoch % 10 == 0:
            img_grid_TB = torchvision.utils.make_grid(torch.cat((data, x_hat), 0).detach().cpu())
            writer.add_image('True (or sampled) image and recon_image', img_grid_TB, OLepoch)
 Counter = 0
 for k in np.arange(0.10,10,0.01):
  Counter += 1
  ##-- Create a standard MVN
  mean = torch.zeros(args.nzg).to(device)
  scale = torch.ones(args.nzg).to(device)
  mvns = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(scale).view(1, args.nzg, args.nzg))

  ##-- Create the proposal, i.e Multivariate Normal with mean = z and CovMatrix = 0.01
  mean = z.view([-1,args.nzg]).to(device)
  scale = k*torch.exp(logvar_first.view([args.nzg])).to(device)
  #scale = 0.01*torch.ones(args.nzg).to(device)
  mvnz = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(scale).view(1, args.nzg, args.nzg))
  sample_shape = torch.Size([])

  likelihood_sample_final = 0
  ## sample and compute
  S = 1
  for iter in range(0,S):
   sample = mvnz.sample(sample_shape)
   log_pz = mvns.log_prob(sample)
   log_rzx = mvnz.log_prob(sample)

   pz = torch.exp(log_pz)
   rzx = torch.exp(log_rzx)

   ##-- Create the proposal, i.e Multivariate Normal with mean = z and CovMatrix = 0.01
   mean = netG(sample.view(-1,args.nzg,1,1)).view(-1,args.imageSize*args.imageSize).to(device)
   scale = torch.exp(logsigmaG).to(device)
   mvnx = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(scale).view(1, args.imageSize**2, args.imageSize**2))
   log_pxz = mvnx.log_prob(data.view(-1,args.imageSize*args.imageSize))
   pxz = torch.exp(log_pxz)

   print('Likelihood: pxz = %.8f (%.8f) .. pz = %.8f (%.8f) .. rzx = %.8f (%.8f)' % (pxz, log_pxz, pz, log_pz, rzx, log_rzx))

   #likelihood_sample_final = likelihood_sample_final + (pxz*pz/rzx)

   likelihood_sample_final = likelihood_sample_final + torch.exp(log_pxz + log_pz - log_rzx)
   print(torch.exp(log_pxz + log_pz - log_rzx))

  #pdb.set_trace()

  img_grid_TB = torchvision.utils.make_grid(torch.cat((data.view(args.imageSize,args.imageSize), mean.view(args.imageSize,args.imageSize)), 0).detach().cpu())
  writer.add_image('mean of z', img_grid_TB, iter)

  writer.add_scalar("Likelihood/log_pz", log_pz, Counter)
  writer.add_scalar("Likelihood/log_rzz", log_rzx, Counter)
  writer.add_scalar("Likelihood/log_pxz", log_pxz, Counter)
  writer.add_scalar("Likelihood/k - overdispersing hyperparameter", k, Counter)

  #pdb.set_trace()


 likelihood_sample_final = torch.log(likelihood_sample_final/S)

 writer.flush()
 writer.close()
 return likelihood_sample_final

