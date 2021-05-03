python3 main_TB.py --dataset mnist --model presgan --lambda_ 0.0 --gpu 1 --save_imgs_every 2 \
                            --restrict_sigma 1 --sigma_min 0.001 --sigma_max 0.2 --epochs 50 --seed 2021

#python3 main_TB.py --dataset mnist --model presgan --lambda_ 0.0 --gpu 1 --save_imgs_every 5 \
#                            --restrict_sigma 1 --sigma_min 0.001 --sigma_max 0.3 \
#                            --logsigma_init -0.5 --sigma_lr 0.001 --lrD 0.001 --lrG 0.0001
