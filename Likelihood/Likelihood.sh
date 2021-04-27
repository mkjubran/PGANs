python3 Likelihood_PGANs_v0.py \
          --ckptE            ../../VAEncoder_E_lambda0.0_beta5_lr0.0002/netE_presgan_MNIST_epoch_99.pth \
          --ckptL            ../../Likelihood_lambda0.0_beta10_lr0.0002 \
          --save_Likelihood ../../Likelihood_lambda0.0_beta10_lr0.0002_Images \
          --beta  10 \
          --nz 64 \
          --epochs 100


