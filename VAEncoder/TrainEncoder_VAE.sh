python3 train_VAE_encoder.py \
          --dataset mnist \
          --ckptG            ../../../PresGANs/mnistimageSize32/SaveS2019/VAEncoderDCGANDecoderMSE2019/netVADec_MNIST_epoch_40.pth \
          --ckptE            ../../../PresGANs/S2019/EncoderType2_VAE_MSE_GS2019epoch40 \
          --save_imgs_folder ../../../PresGANs/S2019/EncoderType2_VAE_MSE_GS2019epoch40_Images \
          --beta  10 \
          --nz 100 \
          --lrE 0.0002 \
          --epochs 50 \
          --imageSize 32 \
          --ngf 32 --ndf 32 --ngfg 32 --ncg 1 --nc 1 --lrE 0.0002