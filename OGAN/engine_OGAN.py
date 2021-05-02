import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
import datetime
import pdb

##-- loading get distribution
def dist(args, device, mu, logvar, mean, scale, data, zr):
 imageSize = args.imageSize

 ##-- compute MVN full batch
 mvn = torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(scale).reshape(1, imageSize*imageSize, imageSize*imageSize))
 log_pxz_mvn = mvn.log_prob(data.view(-1,imageSize*imageSize))

 ##-- sample from standard normal distribution
 std = torch.exp(0.5*logvar)
 std_b = torch.eye(std.size(1)).to(device)
 std_c = std.unsqueeze(2).expand(*std.size(), std.size(1))
 std_3d = std_c * std_b
 mvnz = torch.distributions.MultivariateNormal(mu, scale_tril=std_3d)
 pz_normal = torch.exp(mvnz.log_prob(zr))
 return log_pxz_mvn, pz_normal

def get_overlap_loss(args,device,netE,optimizerE,data,netG,scale,ckptOL):
 log_dir = ckptOL+"/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
 writer = SummaryWriter(log_dir)

 running_loss = 0.0
 counter = 0
 train_loss = []
 overlap_loss = 0;
 epoch = 0;
 while (epoch <= args.epochs) and (overlap_loss >= 0):
        epoch +=1
        counter += 1
        optimizerE.zero_grad()
        x_hat, mu, logvar, z, zr = netE(data, netG)
        mean = x_hat.view(-1,args.imageSize*args.imageSize)

        log_pxz_mvn, pz_normal = dist(args, device, mu, logvar, mean, scale, data, zr)

        ##-- definning overlap loss abd backpropagation 
        overlap_loss = -1*(log_pxz_mvn + pz_normal)
        overlap_loss.backward()
        running_loss += overlap_loss.item()
        optimizerE.step()
        train_loss = running_loss / counter
        
        ##-- print training loss
        #if epoch % 5 ==0:
        #   print(f"Train Loss at epoch {epoch}: {train_loss:.4f}")

        ##-- printing only the positive overlap loss (to avoid printing extremely low numbers after training coverage to low positive value)
        if overlap_loss > 0:
            overlap_loss_sample_final = overlap_loss
            writer.add_scalar("Train Loss", overlap_loss, epoch)

        ##-- write to tensorboard
        if epoch % 10 == 0:
            img_grid_TB = torchvision.utils.make_grid(torch.cat((data, x_hat), 0).detach().cpu())
            writer.add_image('True (or sampled) image and recon_image', img_grid_TB, epoch)
 writer.flush()
 writer.close()
 return overlap_loss_sample_final
