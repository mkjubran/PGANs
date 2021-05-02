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

parser.add_argument('--lrOL', type=float, default=0.001, help='learning rate for overlap loss, default=0.001')
parser.add_argument('--OLbatchSize', type=int, default=100, help='Overlap Loss batch size')

parser.add_argument('--dataset', required=True, help=' ring | mnist | stackedmnist | cifar10 ')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--lrE', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta', type=float, default=1, help='beta1 for KLD in ELBO')
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

 #logsigma_init = -1 #initial value for log_sigma_sian
 if logsigma_file != '':
    logsigmaG = torch.load(logsigma_file)
 else:
   print('A valid logsigma_file for a pretrained PGAN generator must be provided')
 return netG, logsigmaG

##-- loading PGAN discriminator model
def load_discriminator(netD,device,ckptD):
 if ckptD != '':
    netD.load_state_dict(torch.load(ckptD))
 else:
   print('A valid ckptD for a pretrained PGAN discriminator must be provided')

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
 mean = torch.zeros(args.batchSize,nz).to(device)
 scale = torch.ones(nz).to(device)
 mvn = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(scale).view(1, nz, nz))
 sample_z_shape = torch.Size([])
 sample_z = mvn.sample(sample_z_shape).view(-1,nz,1,1)
 #pdb.set_trace()
 recon_images = netG(sample_z)
 #pdb.set_trace()
 return recon_images


if __name__ == "__main__":
 ##-- run on the available GPU otherwise CPUs
 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 ##-- preparing folders to save results of Likelihood
 OL_folders(args)

 ##-- loading and spliting datasets
 trainset, testset = load_datasets(data,args,device)

 ##-- loading PGAN generator model with sigma and setting generator training parameters - G1
 netG1 = nets.Generator(args).to(device)
 optimizerG1 = optim.Adam(netG1.parameters(), lr=args.lrG1)
 netG1, logsigmaG1 = load_generator_wsigma(netG1,device,args.ckptG1,args.logsigma_file_G1)

 ##-- loading PGAN generator model with sigma and setting generator training parameters - G2
 netG = nets.Generator(args).to(device)
 optimizerG = optim.Adam(netG.parameters(), lr=args.lrG2)
 netG, logsigmaG = load_generator_wsigma(netG,device,args.ckptG2,args.logsigma_file_G2)

##-- loading PGAN discriminator model and setting discriminator training parameters - D1
 netD1 = nets.Discriminator(args).to(device)
 optimizerD1 = optim.Adam(netD1.parameters(), lr=args.lrD1)
 netD1 = load_discriminator(netD1,device,args.ckptD1)

##-- loading PGAN discriminator model and setting discriminator training parameters - D2
 netD = nets.Discriminator(args).to(device)
 optimizerD2 = optim.Adam(netD.parameters(), lr=args.lrD2)
 netD = load_discriminator(netD,device,args.ckptD2)

 ##-- loading VAE Encoder and setting encoder training parameters - E1
 netE1 = nets.ConvVAE(args).to(device)
 optimizerE1 = optim.Adam(netE1.parameters(), lr=args.lrE1)
 netE1 = load_encoder(netE1,args.ckptE1)

 ##-- loading VAE Encoder and setting encoder training parameters - E2
 netE = nets.ConvVAE(args).to(device)
 optimizerE = optim.Adam(netE.parameters(), lr=args.lrE2)
 netE = load_encoder(netE,args.ckptE2)

 ##-- write to tensor board
 #writer = SummaryWriter(args.ckptL)

 ##-- setting scale and selecting a random test sample
 imageSize = args.imageSize
 scale = 0.01*torch.ones(imageSize**2)
 scale = scale.to(device)
 #i = torch.randint(0, len(testset),(1,1)) ## selection of the index of test image

 #netE.train()
 ##-- get the overlap loss of a mini batch
 overlap_loss = []

 # to estimate running time
 #start = torch.cuda.Event(enable_timing=True)
 #end = torch.cuda.Event(enable_timing=True)

 ##-- define a new encoder netES to find OL per sample (need to kepp the orogonal netE))
 netES = nets.ConvVAE(args).to(device)
 optimizerES = optim.Adam(netES.parameters(), lr=0.001)
 testset= testset.to(device)
 samples_G = sample_from_generator(args,netG)
 for i in range(args.OLbatchSize):
  #start.record()
  # copy weights of netE to netES
  netES.load_state_dict(copy.deepcopy(netE.state_dict()))

  #sample_G = testset[i].view([1,1,imageSize,imageSize])
  sample_G = samples_G[i].view([1,1,imageSize,imageSize])
  overlap_loss_sample = engine_OGAN.get_overlap_loss(args,device,netES,optimizerES,sample_G,netG,scale,args.ckptOL_E2)
  overlap_loss.append(overlap_loss_sample.item())
  print(overlap_loss)

  # to estimate running time per sample
  #end.record()
  #torch.cuda.synchronize()
  #print(start.elapsed_time(end))  # milliseconds

 #writer.flush()
 #writer.close()

 '''
 running_loss = 0.0
 counter = 0
 train_loss = []
 overlap_loss = 0;
 epoch = 0;
 while (epoch <= args.epochs) and (overlap_loss >= 0):
        epoch +=1
        counter += 1
        optimizerE.zero_grad()
        x_hat, mu, logvar, z, zr = netE(data, netG)
        mean = x_hat.view(-1,imageSize*imageSize)

        log_pxz_mvn, pz_normal = dist(args, mu, logvar, mean, scale, data)

        ##-- definning overlap loss abd backpropagation 
        overlap_loss = -1*(log_pxz_mvn + pz_normal)
        overlap_loss.backward()
        running_loss += overlap_loss.item()
        optimizerE.step()
        train_loss = running_loss / counter

        ##-- print training loss
        if epoch % 5 ==0:
           print(f"Train Loss at epoch {epoch}: {train_loss:.4f}")

        ##-- printing only the positive overlap loss (to avoid printing extremely low numbers after training coverage to low positive value)
        if overlap_loss > 0:
           writer.add_scalar("Train Loss", overlap_loss, epoch)

        ##-- write to tensorboard
        if epoch % 10 == 0:
            img_grid_TB = torchvision.utils.make_grid(torch.cat((data, x_hat), 0).detach().cpu())
            writer.add_image('True and recon_image', img_grid_TB, epoch)

 '''


'''
 ##-- write to tensor board
 writer = SummaryWriter(args.ckptE1)

 # a list to save all the reconstructed images in PyTorch grid format
 grid_images = []

 train_loss = []
 valid_loss = []
 for epoch in range(args.epochs):
    print(f"Epoch {epoch+1} of {args.epochs}")

    train_epoch_loss, elbo, KLDcf, reconloss = train_encoder(
        netE, args, trainset, device, optimizerE, netG, logsigmaG
    )

    valid_epoch_loss, recon_images = validate_encoder(
        netE, args, testset, device, netG, logsigmaG
    )

    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)

    # save the reconstructed images from the validation loop
    save_image(recon_images.cpu(), f"{args.save_imgs_folder}/output{epoch}.jpg")

    # convert the reconstructed images to PyTorch image grid format
    image_grid = make_grid(recon_images.detach().cpu())

    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {valid_epoch_loss:.4f}")
    writer.add_scalar("elbo/KLDcf", KLDcf, epoch)
    writer.add_scalar("elbo/reconloss", reconloss, epoch)
    writer.add_scalar("elbo/elbo", elbo, epoch)
    writer.add_histogram('distribution centers/enc1', netE.enc1.weight, epoch)
    writer.add_histogram('distribution centers/enc2', netE.enc2.weight, epoch)

    # write images to tensorboard
    img_grid_TB = torchvision.utils.make_grid(recon_images.detach().cpu())
    if epoch % 2 == 0:
        writer.add_image('recon_images', img_grid_TB, epoch)

    torch.save(netE.state_dict(), os.path.join(args.ckptE1,'netE_presgan_MNIST_epoch_%s.pth'%(epoch)))

 writer.flush()
 writer.close()


'''