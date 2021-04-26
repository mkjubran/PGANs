
##https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/

from tqdm import tqdm
import torch 
import torch.nn as nn
import pdb

def measure_elbo(mu, logvar, x, x_hat, z, zr,device, criterion, logsigmaG):
    # Use closed form of KL to compute [log_q(z|x) - log_p(z)] assuming P and q are gaussians
    KLDcf = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


    mean = x_hat.view(-1,64*64)
    scale = torch.exp(logsigmaG)
    scale = scale.to(device)

    ### MVN full batch
    mvn = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(scale).reshape(1, 64*64, 64*64))
    #print([mvn.batch_shape, mvn.event_shape])
    log_pxz_mvn = mvn.log_prob(x.view(-1,64*64))
    #pdb.set_trace()

    '''
    ### Normal full batch
    normal = torch.distributions.Normal(mean, scale.reshape(1, 64*64))
    #print([normal.batch_shape, normal.event_shape])
    diagn = torch.distributions.Independent(normal, 1)
    #print([diagn.batch_shape, diagn.event_shape])
    log_pxz_normal = diagn.log_prob(x.view(-1,64*64))
    pdb.set_trace()

    ### MVN iterate over items in batch
    for cnt in range(x_hat.size()[0]):
       meanG = mean[cnt]
       mvni = torch.distributions.MultivariateNormal(mean[cnt],torch.diag(scale))
       mvni_log_prob = mvni.log_prob(x[cnt,:,:,:].view(64*64))
       if cnt == 0:
          log_pxz_mvni = mvni_log_prob.view(1)
       else:
          log_pxz_mvni = torch.cat((log_pxz_mvni, mvni_log_prob.view(1)),0)
    pdb.set_trace()

    ### Normal iterate over items in batch - use torch.dot()
    for cnt in range(x_hat.size()[0]):
       meanG = mean[cnt]
       normalid = torch.distributions.Normal(mean[cnt],scale)
       normalid_log_prob = normalid.log_prob(x[cnt,:,:,:].view(64*64))
       normalid_log_prob = torch.dot(normalid_log_prob, normalid_log_prob)
       if cnt == 0:
          log_pxz_normali = normalid_log_prob.view(1)
       else:
          log_pxz_normali = torch.cat((log_pxz_normali, normalid_log_prob.view(1)),0)

    ### Normal iterate over items in batch - use **2
    for cnt in range(x_hat.size()[0]):
       meanG = mean[cnt]
       normali = torch.distributions.Normal(mean[cnt],scale)
       normali_log_prob = normali.log_prob(x[cnt,:,:,:].view(64*64))
       normali_log_prob = torch.sum(normali_log_prob**2, dim=-1)
       if cnt == 0:
          log_pxz_normali2 = normali_log_prob.view(1)
       else:
          log_pxz_normali2 = torch.cat((log_pxz_normali2, normali_log_prob.view(1)),0)

    pdb.set_trace()
    '''

    #pdb.set_trace()
    ## compute the expectation - sum(p(z)*log_p(x/z))
    # 1'st reshape the std from [batch size, 16] to [batch size, diag(16,16)]
    std = torch.exp(0.5*logvar) # standard deviation
    std_b = torch.eye(std.size(1)).to(device)
    std_c = std.unsqueeze(2).expand(*std.size(), std.size(1))
    std_3d = std_c * std_b
    mvnz = torch.distributions.MultivariateNormal(mu, scale_tril=std_3d)

    pz_normal = torch.exp(mvnz.log_prob(zr))
    pz_log_pxz_mvn = torch.dot(log_pxz_mvn,pz_normal)
    reconloss = pz_log_pxz_mvn

    beta = 5
    elbo = beta*KLDcf - reconloss
    #pdb.set_trace()
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
        reconstruction, mu, logvar, z, zr = model(data, netG)
        elbo, KLDcf, reconloss= measure_elbo(mu, logvar, data, reconstruction, z, zr, device, criterion, logsigmaG)
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
            reconstruction, mu, logvar, z, zr = model(data, netG)
            elbo, KLDcf, reconloss  = measure_elbo(mu, logvar, data, reconstruction, z, zr, device, criterion, logsigmaG)
            #bce_loss = criterion(reconstruction, data)
            #loss = final_loss(bce_loss, mu, logvar)
            loss = elbo
            running_loss += loss.item()
        
            # save the last batch input and output of every epoch
            if i == int(len(dataset)/dataloader.batch_size) - 1:
                recon_images = reconstruction
    val_loss = running_loss / counter
    return val_loss, recon_images


