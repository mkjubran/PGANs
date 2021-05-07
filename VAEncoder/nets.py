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
            nn.ConvTranspose2d(args.ngfg * 2,    args.ngfg, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngfg),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    args.ngfg,      args.ncg, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.main(input)
        return output


# define a Conv VAE
class ConvVAE(nn.Module):
    def __init__(self,args):
        super(ConvVAE, self).__init__()
 
        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=args.nc, out_channels=init_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc3 = nn.Conv2d(
            in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.enc4 = nn.Conv2d(
            in_channels=init_channels*4, out_channels=64, kernel_size=kernel_size, 
            stride=2, padding=0
        )
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, args.nz)
        self.fc_log_var = nn.Linear(128, args.nz)
        self.fc2 = nn.Linear(args.nz, 100)

        # decoder 
        self.dec1 = nn.ConvTranspose2d(
            in_channels=100, out_channels=init_channels*16, kernel_size=kernel_size, 
            stride=1, padding=0
        )
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels*16, out_channels=init_channels*8, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.dec5 = nn.ConvTranspose2d(
            in_channels=init_channels*2, out_channels=args.nc, kernel_size=kernel_size, 
            stride=2, padding=1
        )
    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
 
    def forward(self, x, netG):
        # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        zr = self.reparameterize(mu, log_var)
        z = self.fc2(zr)
        #pdb.set_trace()
        z = z.view(-1, 100, 1, 1)
 
        # decoding using PGAN
        x = netG(z)
        reconstruction = x #torch.sigmoid(x)
        return reconstruction, mu, log_var , z, zr


# define a Conv VAE
class ConvVAEReduced(nn.Module):
    def __init__(self,args):
        super(ConvVAEReduced, self).__init__()
 
        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=args.nc, out_channels=init_channels*8, kernel_size=8, 
            stride=6, padding=1
        )
        #self.enc1 = nn.Conv2d(
        #    in_channels=args.nc, out_channels=init_channels*8, kernel_size=24, 
        #    stride=21, padding=1
        #)
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, args.nz)
        self.fc_log_var = nn.Linear(128, args.nz)
        self.fc2 = nn.Linear(args.nz, 100)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
 
    def forward(self, x, netG):
        # encoding
        x = F.relu(self.enc1(x))
        #pdb.set_trace()
        #x = F.relu(self.enc2(x))
        #pdb.set_trace()
        #x = F.relu(self.enc3(x))
        #pdb.set_trace()
        #x = F.relu(self.enc4(x))
        #pdb.set_trace()
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        zr = self.reparameterize(mu, log_var)
        z = self.fc2(zr)
        #pdb.set_trace()
        z = z.view(-1, 100, 1, 1)
 
        # decoding using PGAN
        x = netG(z)
        reconstruction = x #torch.sigmoid(x)
        return reconstruction, mu, log_var , z, zr
