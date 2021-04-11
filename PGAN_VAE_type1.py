from torch import nn
import torch
import pdb

class Network(torch.nn.Module):
    def __init__(self, dat, netG, nz, ngf, nc):
        super().__init__()

        # output = floor[(width + 2xp - (k-1) -1)/s + 1]
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64  --> output = floor[(64 + 2x1 - (4-1) -1)/2 + 1] = 32
            nn.Conv2d(nc, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32 --> output = floor[(32 + 2x1 - (4-1) -1)/2 + 1] = 16 
            nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16  --> output = floor[(16 + 2x1 - (4-1) -1)/2 + 1] = 8
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8   --> output = floor[(8 + 2x1 - (4-1) -1)/2 + 1] = 4
            nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4   --> output = floor[(4 + 0x1 - (4-1) -1)/1 + 1] = 1
            nn.Conv2d(ngf * 8, nz, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(nz),
            nn.ReLU(True),
            nn.Flatten(),
        )

        
        # distribution parameters
        self.fc_mu = nn.Linear(nz, nz)
        self.fc_var = nn.Linear(nz, nz)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def forward(self, input):
        output = self.main(input)
        mu = self.fc_mu(output)
        var = self.fc_var(output)
        return output, mu, var

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
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

        # encode x to get the mu and variance parameters
        x_encoded = self.main(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)
        pdb.set_trace()

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        pdb.set_trace()

        # decoded
        x_hat = self.decoder(z)
        pdb.set_trace()

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
        pdb.set_trace()

        # kl
        kl = self.kl_divergence(z, mu, std)
        pdb.set_trace()

        # elbo
        elbo = (kl - recon_loss)
        pdb.set_trace()

        elbo = elbo.mean()
        pdb.set_trace()

        self.log_dict({
            'elbo': elbo,
            'kl': kl.mean(),
            'recon_loss': recon_loss.mean(),
            'reconstruction': recon_loss.mean(),
            'kl': kl.mean(),
        })

        return elbo
