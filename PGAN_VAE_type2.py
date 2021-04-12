## https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/

from torch import nn
import torch
import pdb
import torch.nn.functional as F

class Network(torch.nn.Module):
    def __init__(self, dat, netG, nz, ngf, nc):
        super().__init__()

        kernel_size = 4 # (4, 4) kernel
        init_channels = 8 # initial number of filters
        image_channels = 1 # MNIST images are grayscale
        latent_dim = nz # latent dimension for sampling

        # encoder
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, 
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
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 100)


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
            in_channels=init_channels*2, out_channels=image_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )



    def decoding(self, zd):
        # decoding
        #pdb.set_trace()
        x = F.relu(self.dec1(zd))
        #pdb.set_trace()
        x = F.relu(self.dec2(x))
        #pdb.set_trace()
        x = F.relu(self.dec3(x))
        #pdb.set_trace()
        x = F.relu(self.dec4(x))
        reconstruction = torch.sigmoid(self.dec5(x))
        #pdb.set_trace()
        return reconstruction

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
 
    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        #pdb.set_trace()
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        #pdb.set_trace()
        z = self.fc2(z)
        #pdb.set_trace()
        z = z.view(-1, 100, 1, 1)

        x = F.relu(self.dec1(z))
        #pdb.set_trace()
        x = F.relu(self.dec2(x))
        #pdb.set_trace()
        x = F.relu(self.dec3(x))
        #pdb.set_trace()
        x = F.relu(self.dec4(x))
        reconstruction = torch.sigmoid(self.dec5(x))
 
        return z, mu, log_var, reconstruction

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        #pdb.set_trace()
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        #pdb.set_trace()

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)
        #pdb.set_trace()

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        #pdb.set_trace()

        return kl


    def PGAN_ELBO(self, z, mu, std, x_hat, logscale, x):
        # --------------------------
        # Compute second part of Eq 20 in PresGAN paper
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # 3. compute probability p(z) and p(z/x)
        log_qzx_sum = log_qzx.sum(-1)
        log_pz_sum = log_pz.sum(-1)

        # 4. measure prob of seeing image under p(x|z)
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)
        log_pxz = dist.log_prob(x)
        log_pxz_sum = log_pxz.sum(dim=(1, 2, 3))

        # 5. compute ELBO
        ELBO = log_pxz_sum + log_pz_sum - log_qzx_sum

        return log_qzx_sum, log_pz_sum, log_pxz_sum, ELBO

    def training_step(self, batch, batch_idx):
        x, _ = batch

        # encode x to get the mu and variance parameters and sample z from q distribution
        z, mu, log_var, x_hat = self.forward(x)
        #pdb.set_trace()

        # decoded
        #x_hat = self.decoder(z)
        #pdb.set_trace()

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
        #pdb.set_trace()

        # kl
        kl = self.kl_divergence(z, mu, std)
        #pdb.set_trace()

        # elbo
        elbo = (kl - recon_loss)
        #pdb.set_trace()

        elbo = elbo.mean()
        #pdb.set_trace()

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean(),
            'reconstruction': recon_loss.mean(),
            'kl': kl.mean(),
        })

        return elbo
