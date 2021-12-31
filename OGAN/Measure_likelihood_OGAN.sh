Path1='mnistimageSize32'
Path2='tvGAN_mnistimSize32_lambda0.0_lr0.0002_W10.0_W20.0_valbatches100_S2000_GS2019_GS2020'
fname='mnist'
FNAME='MNIST'
VAE_epoch='epoch_20' #mnist 20, cifar 180
VAEepoch='epoch20' #mnist 20, cifar 180
dataset='mnist'

#Path1='cifar10imageSize32'
#Path2='tvGAN_cifar10imSize32_lambda0.0_lr0.0002_W10.0_W20.0_valbatches100_S2000_GS2019_GS2020'
#fname='cifar10'
#FNAME='CIFAR10'
#VAE_epoch='epoch_180' #mnist 20, cifar 180
#VAEepoch='epoch180' #mnist 20, cifar 180
#dataset='cifar10'

#Path1='celebaimageSize32'
#Path2='tvGAN_celebaimSize32_lambda0.0_lr0.0002_W10.0_W20.0_valbatches100_S2000_GS2019_GS2020'
#fname='celeba'
#FNAME='CELEBA'
#VAE_epoch='epoch_150'
#VAEepoch='epoch180'
#dataset='celeba'

Step='3000'

Path='../../../PresGANs/'$Path1'/'$Path2'/G'
python3 measure_likelihood_batch.py \
          --dataset $dataset \
          --sample_from dataset \
          --ckptG1            $Path'/netG1_presgan_'$fname'_step_'$Step'.pth' \
          --logsigma_file_G1  $Path'/log_sigma_G1_'$fname'_step_'$Step'.pth' \
          --ckptD1            $Path'/netD1_presgan_'$fname'_step_'$Step'.pth' \
          --ckptE1            '../../../PresGANs/'$fname'imageSize32/SaveS2019/EncoderType2_lambda0.0_GS2019'$VAEepoch'/netE_presgan_'$FNAME'_'$VAE_epoch'.pth'\
          --ckptG2            $Path'/netG2_presgan_'$fname'_step_'$Step'.pth' \
          --logsigma_file_G2  $Path'/log_sigma_G2_'$fname'_step_'$Step'.pth' \
          --ckptD2            $Path'/netD2_presgan_'$fname'_step_'$Step'.pth' \
          --ckptE2            '../../../PresGANs/'$fname'imageSize32/SaveS2020/EncoderType2_lambda0.0_GS2020'$VAEepoch'/netE_presgan_'$FNAME'_'$VAE_epoch'.pth'\
          --save_likelihood_folder    '../../../PresGANs/MeasureLL_'$Path2'_ImSampling_GScaleOne/Step'$Step'Repeat'\
          --number_samples_likelihood 10000\
          --S 10000 \
          --seed_G1 2019 \
          --seed_G2 2020 \
          --lambda_ 0.0 \
          --W1 0.0 \
          --W2 0.0 \
          --lrOL 0.0002 \
          --beta  10 \
          --nz 100 \
          --OLepochs 10000 \
          --epochs 50 \
          --batchSize 100\
          --OLbatchSize 100\
          --num_gen_images 20\
          --GPU 0 \
          --overlap_loss_min -10000000000 \
          --overdispersion 1.2 \
          --imageSize 32 --ngf 32 --ndf 32 --ngfg 32 --ndfg 32 --ncg 1 --nc 1 --lrE1 0.0002 --lrE2 0.0002

