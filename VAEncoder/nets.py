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
            nn.ConvTranspose2d(args.ngfg * 2,    args.ncg, 4, 2, 1, bias=False),
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
        #self.fc2 = nn.Linear(args.nz, 100)

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
 
    def forward(self, x, args):
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
        #z = self.fc2(zr)
        #pdb.set_trace()
        z = zr.view(-1, 100, 1, 1)

        # decoding using PGAN
        #x = netG(z)
        #reconstruction = x #torch.sigmoid(x)
        return mu, log_var , z, zr

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
        #self.fc2 = nn.Linear(args.nz, 100)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
 
    def forward(self, x, args):
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
        #z = self.fc2(zr)
        #pdb.set_trace()
        z = zr.view(-1, 100, 1, 1)
 
        # decoding using PGAN
        #x = netG(z)
        #reconstruction = x #torch.sigmoid(x)
        return mu, log_var , z, zr


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

## VAE based on FC layers only
class LinearVAEncoder(nn.Module):
    def __init__(self,args):
        super(LinearVAEncoder, self).__init__()
        x_dim=args.imageSize**2
        h_dim1=int((args.imageSize**2)/2)
        h_dim2=int((args.imageSize**2)/4)
        z_dim=16

        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)

    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
      
    def forward(self, x, args):
        mu, log_var = self.encoder(x.view(-1, args.imageSize**2))
        z = self.sampling(mu, log_var)
        return mu, log_var, z, z


class LinearVADecoder(nn.Module):
    def __init__(self,args):
        super(LinearVADecoder, self).__init__()
        x_dim=args.imageSize**2
        h_dim1=int((args.imageSize**2)/2)
        h_dim2=int((args.imageSize**2)/4)

        # decoder part
        self.dec1 = nn.Linear(args.nz, h_dim2)
        self.dec2 = nn.Linear(h_dim2, h_dim1)
        self.dec3 = nn.Linear(h_dim1, x_dim)

    def decoder(self, z):
        x = F.relu(self.dec1(z))
        x = F.relu(self.dec2(x))
        x = torch.sigmoid(self.dec3(x))
        return x

    def forward(self, z, args):
        batch, _, _, _ = z.shape
        z = z.view(batch, -1)
        reconstruction = self.decoder(z).view(-1,1,args.imageSize,args.imageSize)
        return reconstruction

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
            nn.ConvTranspose2d(args.ngfg * 2,    args.ngfg, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngfg),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    args.ngfg,      args.ncg, 4, 2, 1, bias=False),
            #nn.Tanh()
            nn.Sigmoid() # replace the Tanh() of DCGAN, this is required by nn.bce_loss to make sure the output i between 0 and 1
            # state size. (nc) x 64 x 64
        )

    def forward(self, input, args):
        output = self.main(input)
        return output
