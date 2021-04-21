
##https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/

from tqdm import tqdm
import torch 
import torch.nn as nn
import pdb

def measure_elbo(mu, logvar, x, x_hat, z, device, criterion, logsigmaG):
    # Use closed form of KL to compute [log_q(z|x) - log_p(z)] assuming P and q are gaussians
    KLDcf = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # measure prob of seeing image under log_p(x|z)
    #logscale = nn.Parameter(torch.Tensor([0.0]))
    logscale = logsigmaG.view(1,1,64,64)
    scale = torch.exp(logscale)
    scale = scale.repeat(100, 1, 1, 1)
    mean = x_hat
    scale = scale.to(device)
    dist = torch.distributions.Normal(mean, scale) 
    #pdb.set_trace()
    log_pxz = dist.log_prob(x)
    #log_pxz = log_pxz/log_pxz.max()

    # measure elbo using log_pxz ==> elbo = [log_q(z|x) - log_p(z) - log_p(x|z)] = [KLD - log_q(x|z)] 
    reconloss = log_pxz.sum(dim=(0,1,2,3))
    #pdb.set_trace()
    elbo = KLDcf - 0.1*reconloss

    # measure elbo using MSE construction loss ==> elbo = [log_q(z|x) - log_p(z) - ReconLoss] = [KLD - ReconLoss] 
    #reconloss = criterion(x_hat,x) # BCE (x_hat,x) or MSE(x_hat,x)
    #elbo = KLDcf + reconloss
    return elbo, KLDcf, reconloss

def train_PGAN(model, dataloader, dataset, device, optimizer, criterion, netG, logsigmaG):
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar, z = model(data, netG)
        elbo, KLDcf, reconloss= measure_elbo(mu, logvar, data, reconstruction, z, device, criterion, logsigmaG)
        #bce_loss = criterion(reconstruction, data)
        #loss = final_loss(bce_loss, mu, logvar)
        loss = elbo
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss / counter 
    return train_loss, elbo, KLDcf, reconloss

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
model, dataloader, dataset, device, criterion, netG, logsigmaG):
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
            counter += 1
            data= data[0]
            data = data.to(device)
            reconstruction, mu, logvar, z = model(data, netG)
            elbo, KLDcf, reconloss  = measure_elbo(mu, logvar, data, reconstruction, z, device, criterion, logsigmaG)
            #bce_loss = criterion(reconstruction, data)
            #loss = final_loss(bce_loss, mu, logvar)
            loss = elbo
            running_loss += loss.item()
        
            # save the last batch input and output of every epoch
            if i == int(len(dataset)/dataloader.batch_size) - 1:
                recon_images = reconstruction
    val_loss = running_loss / counter
    return val_loss, recon_images


