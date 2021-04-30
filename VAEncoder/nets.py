import torch.nn as nn

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

class Discriminator(nn.Module):
    def __init__(self, imgSize, ndf, nc, args):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, args.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(args.ndf, args.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(args.ndf * 2, args.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, args.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(args.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)


class Encoder(nn.Module):
    def __init__(self, imgSize, nz, ngf, nc, args):
        super(Encoder, self).__init__()
        
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(args.nc, args.ngf, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.Conv2d(args.ngf, args.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.Conv2d(args.ngf * 2, args.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.Conv2d(args.ngf * 4, args.ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.Conv2d(args.ngf * 8, args.nz, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.nz),
            nn.ReLU(True),

        )

    def forward(self, input):
        output = self.main(input)
        return output


class VAEncoder(nn.Module):
    def __init__(self, imgSize, nz, ngf, nc, args):
        super(VAEncoder, self).__init__()
        '''
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ngf, 4, 2, 1, bias=False),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.Conv2d(ngf * 8, nz, 4, 1, 0, bias=False),
            nn.BatchNorm2d(nz),
            nn.ReLU(True),
            nn.Flatten(),
        )
        '''

        self.main = nn.Sequential(
           nn.Conv2d(args.nc, args.ngf, 4, 2, 1, bias=False),
           nn.Conv2d(args.ngf, args.ngf, 4, 2, 1, bias=False),
        )

        # distribution parameters
        self.fc_mu = nn.Linear(args.nz, args.nz)
        self.fc_var = nn.Linear(args.nz, args.nz)

    def forward(self, input):
        output = self.main(input)
        mu = self.fc_mu(output)
        var = self.fc_var(output)
        return mu, var

