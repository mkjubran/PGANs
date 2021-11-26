#python3 main_TB.py --dataset mnist --model presgan --lambda_ 0.0 --gpu 0 --save_imgs_every 2 \
#                            --restrict_sigma 1 --sigma_min 0.001 --sigma_max 0.2 --epochs 50 --seed 2021

python3 main_TB.py --dataset mnist --model presgan --lambda_ 0.0 --gpu 0 --save_imgs_every 2 --imageSize 32 --ngf 32 --ndf 32\
                            --restrict_sigma 1 --sigma_min 0.001 --sigma_max 0.2 --epochs 50 --seed 2020\
                            --lrD 0.0002 --lrG 0.0002 --sigma_lr 0.001 --lrE 0.0002 --logsigma_init -0.5

#python3 main_TB.py --dataset celeba --model presgan --lambda_ 0.0 --gpu 0 --save_imgs_every 2 --imageSize 64 --ngf 64 --ndf 64\
#                            --restrict_sigma 1 --sigma_min 0.001 --sigma_max 0.3 --epochs 200 --seed 2020\
#                            --lrD 0.0002 --lrG 0.0002 --sigma_lr 0.001 --lrE 0.0002 --logsigma_init -0.5

