python3 measure_likelihood_batch.py \
          --dataset mnist \
          --sample_from dataset \
          --ckptG1            ../../../PresGANs/mnistimageSize32/SaveS2019/presgan_lambda_0.0_GS2019/netG_presgan_mnist_epoch_20.pth \
          --logsigma_file_G1  ../../../PresGANs/mnistimageSize32/SaveS2019/presgan_lambda_0.0_GS2019/log_sigma_mnist_20.pth \
          --ckptD1            ../../../PresGANs/mnistimageSize32/SaveS2019/presgan_lambda_0.0_GS2019/netD_presgan_mnist_epoch_20.pth \
          --ckptE1            ../../../PresGANs/mnistimageSize32/SaveS2019/VAEncoderType2_lambda0.0_GS2019epoch20/netE_presgan_MNIST_epoch_19.pth\
          --ckptG2            ../../../PresGANs/mnistimageSize32/SaveS2020/presgan_lambda_0.0_GS2020/netG_presgan_mnist_epoch_20.pth \
          --logsigma_file_G2  ../../../PresGANs/mnistimageSize32/SaveS2020/presgan_lambda_0.0_GS2020/log_sigma_mnist_20.pth \
          --ckptD2            ../../../PresGANs/mnistimageSize32/SaveS2020/presgan_lambda_0.0_GS2020/netD_presgan_mnist_epoch_20.pth \
          --ckptE2            ../../../PresGANs/mnistimageSize32/SaveS2020/VAEncoderType2_lambda0.0_GS2020epoch20/netE_presgan_MNIST_epoch_19.pth\
          --save_likelihood_folder    ../../../PresGANs/MeasureLL__mnistimSize32_lambda0.0_lr0.0002_W10.0_W20.0_valbatches100_S2000_GS2019_GS2020 \
          --number_samples_likelihood 2000\
          --S 2000 \
          --seed_G1 2019 \
          --seed_G2 2020 \
          --lambda_ 0.0 \
          --W1 0.0 \
          --W2 0.0 \
          --lrOL 0.0002 \
          --beta  10 \
          --nz 100 \
          --OLepochs 50000 \
          --epochs 50 \
          --batchSize 20\
          --OLbatchSize 20\
          --num_gen_images 20\
          --GPU 0 \
          --overlap_loss_min -10000000000 \
          --imageSize 32 --ngf 32 --ndf 32 --ngfg 32 --ndfg 32 --ncg 1 --nc 1 --lrE1 0.0002 --lrE2 0.0002
