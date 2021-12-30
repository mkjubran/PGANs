Path1='mnistimageSize32'
dataset='mnist'

#Path1='cifar10imageSize32'
#dataset='cifar10'

#Path1='celebaimageSize32'
#dataset='celeba'

epoch='49'
Seed='2019'
Path2='VAEncoderDCGANDecoderBCE'$Seed

python3 measure_likelihood_batch_VAE.py \
          --dataset $dataset \
          --sample_from dataset \
          --ckptDec            '../../../PresGANs/'$Path1'/SaveS'$Seed'/'$Path2'/netVADec_MNIST_epoch_'$epoch'.pth' \
          --ckptE            '../../../PresGANs/'$Path1'/SaveS'$Seed'/'$Path2'/netVAEnc_MNIST_epoch_'$epoch'.pth' \
          --save_likelihood_folder   '../../../PresGANs/MeasureLL_VAE_HMC_'$Path2'_ImSampling/Epoch'$epoch\
          --number_samples_likelihood 2000\
          --lambda_ 0.0 \
          --lrOL 0.0002 \
          --beta  10 \
          --nz 100 \
          --OLepochs 1000 \
          --epochs 50 \
          --batchSize 100\
          --OLbatchSize 100\
          --num_gen_images 20\
          --seed_VAE 2019 \
          --S 10000 \
          --GPU 0 \
          --overlap_loss_min -10000000000 \
          --valevery 600 \
          --valbatches 600 \
          --overdispersion 1.0 \
          --imageSize 32 --ngf 32 --ndf 32 --ngfg 32 --ndfg 32 --ncg 1 --nc 1 --lrE 0.0002
