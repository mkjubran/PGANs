from tqdm import tqdm
import torch
import torch.nn as nn
import pdb

def measure_elbo(args, mu, logvar, x, x_hat, z, zr,device, logsigmaG):
    imageSize = args.imageSize
    # Use closed form of KL to compute [log_q(z|x) - log_p(z)] assuming P and q are gaussians
    KLDcf = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    mean = x_hat.view(-1,imageSize*imageSize)
    scale = torch.exp(logsigmaG)
    scale = scale.to(device)

    ### MVN full batch
    mvn = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(scale).reshape(1, imageSize*imageSize, imageSize*imageSize))
    log_pxz_mvn = mvn.log_prob(x.view(-1,imageSize*imageSize))

    '''
    ### Normal full batch
    normal = torch.distributions.Normal(mean, scale.reshape(1, 64*64))
    #print([normal.batch_shape, normal.event_shape])
    diagn = torch.distributions.Independent(normal, 1)
    #print([diagn.batch_shape, diagn.event_shape])
    log_pxz_normal = diagn.log_prob(x.view(-1,64*64))
    pdb.set_trace()

    '''

    ## compute the expectation - sum(p(z)*log_p(x/z))
    # 1'st reshape the std from [batch size, 16] to [batch size, diag(16,16)]
    std = torch.exp(0.5*logvar) # standard deviation
    std_b = torch.eye(std.size(1)).to(device)
    std_c = std.unsqueeze(2).expand(*std.size(), std.size(1))
    std_3d = std_c * std_b
    mvnz = torch.distributions.MultivariateNormal(mu, scale_tril=std_3d)

    pz_normal = torch.exp(mvnz.log_prob(zr))
    if args.nc == 1:
       pz_log_pxz_mvn = torch.dot(log_pxz_mvn,pz_normal)
    elif args.nc == 3:
       pz_log_pxz_mvn = torch.dot(log_pxz_mvn[0::3],pz_normal)+torch.dot(log_pxz_mvn[1::3],pz_normal)+torch.dot(log_pxz_mvn[2::3],pz_normal)
    else:
       print('Number of channels in the source are not supported')
    reconloss = pz_log_pxz_mvn

    beta = args.beta
    elbo = beta*KLDcf - reconloss

    return elbo, KLDcf, reconloss

def train_encoder(netE, args, X_training, device, optimizer, netG, logsigmaG):
    netE.train()
    running_loss = 0.0
    counter = 0
    for i in tqdm(range(0, len(X_training), args.batchSize)):
        stop = min(args.batchSize, len(X_training[i:]))
        data = X_training[i:i+stop].to(device)
        counter += 1
        optimizer.zero_grad()

        #reconstruction, mu, logvar, z, zr = netE(data, netG)
        mu, logvar, z, zr = netE(data, args)
        reconstruction = netG(z)

        elbo, KLDcf, reconloss= measure_elbo(args, mu, logvar, data, reconstruction, z, zr, device, logsigmaG)
        loss = elbo
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss / counter
    return train_loss, elbo, KLDcf, reconloss

def validate_encoder(netE, args, X_testing, device, netG, logsigmaG):
    netE.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i in tqdm(range(0, len(X_testing), args.batchSize)):
            stop = min(args.batchSize, len(X_testing[i:]))
            data = X_testing[i:i+stop].to(device)
            counter += 1

            #reconstruction, mu, logvar, z, zr = netE(data, netG)
            mu, logvar, z, zr = netE(data, args)
            reconstruction = netG(z)

            elbo, KLDcf, reconloss  = measure_elbo(args, mu, logvar, data, reconstruction, z, zr, device, logsigmaG)
            loss = elbo
            running_loss += loss.item()
       
            # save the last batch input and output of every epoch
            if (i+stop) == X_testing.size(0):
                recon_images = reconstruction
    val_loss = running_loss / counter
    return val_loss, recon_images


def train_encoder_decoder(netE, args, X_training, device, optimizer, criterion, netDec, optimizerDec):
    netE.train()
    netDec.train()
    running_loss = 0.0
    counter = 0
    for i in tqdm(range(0, len(X_training), args.batchSize)):
        stop = min(args.batchSize, len(X_training[i:]))
        data = X_training[i:i+stop].to(device)

        counter += 1
        optimizer.zero_grad()
        optimizerDec.zero_grad()

        #reconstruction, mu, logvar, z, zr = netE(data, args)
        mu, logvar, z, zr = netE(data, args)
        reconstruction = netDec(z, args)

        #k=10
        #logscale = k*torch.ones(args.imageSize**2).to(device)
        #elbo, KLDcf, reconloss= measure_elbo(args, mu, logvar, data, reconstruction, z, zr, device, logscale)
        #loss = elbo

        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)

        loss.backward()
        running_loss += loss.item()
        optimizer.step()
        optimizerDec.step()

        if (i+stop) == X_training.size(0):
            recon_images = reconstruction

    train_loss = running_loss / counter
    return train_loss, recon_images

def validate_encoder_decoder(netE, args, X_testing, device, criterion, netDec, optimizerDec):
    netE.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i in tqdm(range(0, len(X_testing), args.batchSize)):
            stop = min(args.batchSize, len(X_testing[i:]))
            data = X_testing[i:i+stop].to(device)
            counter += 1

            #reconstruction, mu, logvar, z, zr = netE(data, args)
            mu, logvar, z, zr = netE(data, args)
            reconstruction = netDec(z, args)
   
            #k=10
            #logscale = 1*torch.ones(args.imageSize**2).to(device)
            #elbo, KLDcf, reconloss  = measure_elbo(args, mu, logvar, data, reconstruction, z, zr, device, logscale)
            #loss = elbo

            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
       
            running_loss += loss.item()
      
            # save the last batch input and output of every epoch
            if (i+stop) == X_testing.size(0):
                recon_images = reconstruction

    val_loss = running_loss / counter
    return val_loss, recon_images


def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
