
##https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/

from tqdm import tqdm
import torch 
import torch.nn as nn
import pdb

def measure_elbo(mu, logvar, x, x_hat, z, device, criterion):
    # Use closed form of KL to compute [log_q(z|x) - log_p(z)] assuming P and q are gaussians
    KLDcf = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # measure prob of seeing image under log_p(x|z)
    logscale = nn.Parameter(torch.Tensor([0.0]))
    scale = torch.exp(logscale)
    mean = x_hat
    scale = scale.to(device)
    dist = torch.distributions.Normal(mean, scale) 
    log_pxz = dist.log_prob(x)

    # measure log_q(z|x)
    std = torch.exp(logvar / 2)
    #try:
    q = torch.distributions.Normal(mu, std)
    #except:
    #   #pdb.set_trace()
    #   std[std==0] = 1
    #   q = torch.distributions.Normal(mu, std)
    log_qzx = q.log_prob(z)

    # measure log_p(z)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    log_pz = p.log_prob(z)

    # measure mean of log_p(x|z), log_q(z|x), log_p(z)
    pdb.set_trace()
    log_pxz = log_pxz.sum(dim=(1,2,3))

    ## measure KLD through sampling
    KLDsample = log_qzx - log_pz
    KLDsample = KLDsample.sum(-1)   # sum over last dim to go from single dim distribution to multi-dim

    # measure elbo = [log_p(x|z) + log_p(z) - log_q(z|x)]
    elbo = log_pxz - KLDcf # elbo os VAE

    # Construction Loss [for testing only]
    #bce_loss = criterion(x_hat,x)
    #pdb.set_trace()
    #elbo = bce_loss + KLDcf #here elbo is the loss and not the VAE elbo
    pdb.set_trace()
    return elbo, log_pxz, KLDsample, KLDcf

def train_PGAN(model, dataloader, dataset, device, optimizer, criterion, netG):
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar, z = model(data, netG)
        elbo, log_pxz, KLDsample, KLDcf = measure_elbo(mu, logvar, data, reconstruction, z, device, criterion)
        #bce_loss = criterion(reconstruction, data)
        #loss = final_loss(bce_loss, mu, logvar)
        loss = elbo
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss / counter 
    return train_loss, elbo, log_pxz, KLDsample, KLDcf

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

def train(model, dataloader, dataset, device, optimizer, criterion):
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar, z = model(data)
        bce_loss = criterion(reconstruction, data)
        loss = final_loss(bce_loss, mu, logvar)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss / counter 
    return train_loss

def validate(
model, dataloader, dataset, device, criterion, netG):
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
            counter += 1
            data= data[0]
            data = data.to(device)
            reconstruction, mu, logvar, z = model(data, netG)
            elbo, log_pxz, KLDsample, KLDcf  = measure_elbo(mu, logvar, data, reconstruction, z, device, criterion)
            #bce_loss = criterion(reconstruction, data)
            #loss = final_loss(bce_loss, mu, logvar)
            loss = elbo
            running_loss += loss.item()
        
            # save the last batch input and output of every epoch
            if i == int(len(dataset)/dataloader.batch_size) - 1:
                recon_images = reconstruction
    val_loss = running_loss / counter
    return val_loss, recon_images


