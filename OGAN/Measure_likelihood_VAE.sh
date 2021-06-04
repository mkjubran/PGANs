python3 measure_likelihood_VAE.py \
          --dataset mnist \
          --sample_from generator \
          --ckptDec                  ../../../PresGANs/S2019/VAEncoderDecoder2019/netVADec_MNIST_epoch_20.pth \
          --ckptE                  ../../../PresGANs/S2019/VAEncoderDecoder2019/netVAEnc_MNIST_epoch_20.pth \
          --save_likelihood_folder    ../../../PresGANs/Likelihood/True0.0_OLoss_VADec2019/Likelihood_EType2 \
          --number_samples_likelihood 1000\
          --lambda_ 0 \
          --lrOL 0.0001 \
          --beta  10 \
          --nz 100 \
          --OLepochs 2000 \
          --epochs 500 \
          --batchSize 100\
          --OLbatchSize 4\
          --num_gen_images 100


