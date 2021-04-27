##https://debuggercafe.com/convolutional-variational-autoencoder-in-pytorch-on-mnist-dataset/

import imageio
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image
to_pil_image = transforms.ToPILImage()
def image_to_vid(images,args):
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave(args.save_imgs_folder+'/generated_images.gif', imgs)
def save_reconstructed_images(recon_images, epoch,args):
    save_image(recon_images.cpu(), f"{args.save_imgs_folder}/output{epoch}.jpg")
def save_loss_plot(train_loss, valid_loss,args):
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validataion loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(args.save_imgs_folder+'/loss.jpg')
    plt.show()
def save_reconstructed_images_SG(recon_images, epoch,savefolder):
    save_image(recon_images.cpu(), '%s/output_SG_Sample%03d.png' % (savefolder, epoch))
    #save_image(recon_images.cpu(), f"{savefolder}/Output_SG_{epoch}.jpg")

