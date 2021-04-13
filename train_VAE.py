##https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/

import torch
import torch.optim as optim
import torch.nn as nn
import model
import torchvision.transforms as transforms
import torchvision
import matplotlib
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from engine import train, validate
from utils import save_reconstructed_images, image_to_vid, save_loss_plot
from torch.utils.tensorboard import SummaryWriter

import shutil
import os
import pdb

matplotlib.style.use('ggplot')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize the model
model = model.ConvVAE().to(device)

# set the learning parameters
lr = 0.001
epochs = 100
batch_size = 64
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss(reduction='sum')

# a list to save all the reconstructed images in PyTorch grid format
grid_images = []

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

## Checking paths and folders
if not os.path.exists('../outputs'):
    os.makedirs('../outputs')

if not os.path.exists('../log'):
    os.makedirs('../log')
else:
    shutil.rmtree('../log')
    os.makedirs('../log') 

# training set and train data loader
trainset = torchvision.datasets.MNIST(
    root='../input', train=True, download=True, transform=transform
)
trainloader = DataLoader(
    trainset, batch_size=batch_size, shuffle=True
)
# validation set and validation data loader
testset = torchvision.datasets.MNIST(
    root='../input', train=False, download=True, transform=transform
)
testloader = DataLoader(
    testset, batch_size=batch_size, shuffle=False
)

train_loss = []
valid_loss = []

writer = SummaryWriter('../log')

for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, trainloader, trainset, device, optimizer, criterion
    )
    valid_epoch_loss, recon_images = validate(
        model, testloader, testset, device, criterion
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    # save the reconstructed images from the validation loop
    save_reconstructed_images(recon_images, epoch+1)
    # convert the reconstructed images to PyTorch image grid format
    image_grid = make_grid(recon_images.detach().cpu())
    grid_images.append(image_grid)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {valid_epoch_loss:.4f}")


    writer.add_scalar("Train Loss", train_epoch_loss, epoch)
    writer.add_scalar("Val Loss", valid_epoch_loss, epoch)

    # log images to tensorboard
    # create grid of images
    #pdb.set_trace()
    img_grid_TB = torchvision.utils.make_grid(recon_images.detach().cpu())

    # write to tensorboard
    writer.add_image('recon_images', img_grid_TB)

writer.flush()
writer.close()

# save the reconstructions as a .gif file
image_to_vid(grid_images)
# save the loss plots to disk
save_loss_plot(train_loss, valid_loss)
print('TRAINING COMPLETE')



