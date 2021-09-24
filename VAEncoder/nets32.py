import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

kernel_size = 4 # (4, 4) kernel
init_channels = 8 # initial number of filters

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
       
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     args.nzg, args.ngfg * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.ngfg * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(args.ngfg * 8, args.ngfg * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngfg * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(args.ngfg * 4, args.ngfg * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngfg * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(args.ngfg * 2,  args.nc, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(args.ngfg),
            #nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            #nn.ConvTranspose2d(    args.ngfg,      args.ncg, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output

## PressGAN Discriminator
class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(args.nc, args.ndfg, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(args.ndfg, args.ndfg * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndfg * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(args.ndfg * 2, args.ndfg * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndfg * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(args.ndfg * 4, 1, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(args.ndfg * 8),
            #nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            #nn.Conv2d(args.ndfg * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)

# define a Conv VAE Type 2
class ConvVAEType2(nn.Module):
    def __init__(self,args):
        super(ConvVAEType2, self).__init__()

        # encoder
        self.enc1 = nn.Conv2d(args.nc, init_channels, 4, 2, 1, bias=False)
        self.leaky1 = nn.LeakyReLU(0.2, inplace=False)
            # state size. (ndf) x 32 x 32
        self.enc2 = nn.Conv2d(init_channels, init_channels * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(init_channels * 2)
        self.leaky2 = nn.LeakyReLU(0.2, inplace=False)
            # state size. (ndf*2) x 16 x 16
        self.enc3 = nn.Conv2d(init_channels * 2, init_channels * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(init_channels * 4)
        self.leaky3 = nn.LeakyReLU(0.2, inplace=False)
            # state size. (ndf*4) x 8 x 8
        self.enc4 = nn.Conv2d(init_channels * 4, init_channels * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(init_channels * 8)
        self.leaky4 = nn.LeakyReLU(0.2, inplace=False)

        # fully connected layers for learning representations
        self.fc1 = nn.Linear(init_channels * 8, 128)
        self.fc_mu = nn.Linear(128, args.nz)
        self.fc_log_var = nn.Linear(128, args.nz)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
 
    #def forward(self, x, netG):
    def forward(self, x, args):
        # encoding
        x = self.enc1(x)
        x = self.leaky1(x)
        x = self.enc2(x)
        x = self.bn2(x)
        x = self.leaky2(x)
        x = self.enc3(x)
        x = self.bn3(x)
        x = self.leaky3(x)
        x = self.enc4(x)
        x = self.bn4(x)
        x = self.leaky4(x)
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        zr = self.reparameterize(mu, log_var)
        #z = self.fc2(zr)
        #pdb.set_trace()
        z = zr.view(-1, 100, 1, 1)
 
        # decoding using PGAN *******************************<<<<
        #x = netG(z)
        #reconstruction = x #torch.sigmoid(x)
        #return reconstruction, mu, log_var , z, zr
        return mu, log_var , z, zr

class VAEGenerator(nn.Module):
    def __init__(self, args):
        super(VAEGenerator, self).__init__()
        
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     args.nzg, args.ngfg * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.ngfg * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(args.ngfg * 8, args.ngfg * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngfg * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(args.ngfg * 4, args.ngfg * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngfg * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(args.ngfg * 2,    args.ncg, 4, 2, 1, bias=False),
            #nn.BatchNorm2d(args.ngfg),
            #nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            #nn.ConvTranspose2d(    args.ngfg,      args.ncg, 4, 2, 1, bias=False),
            #nn.Tanh()
            nn.Sigmoid() # replace the Tanh() of DCGAN, this is required by nn.bce_loss to make sure the output i between 0 and 1
            # state size. (nc) x 64 x 64
        )

    def forward(self, input, args):
        output = self.main(input)
        return output
