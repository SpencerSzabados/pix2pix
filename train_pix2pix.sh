
# Train pix2pix on lysto64_random_crop_pix2pix
# -----------------------------------------------
# python train.py --model pix2pix --netG unet_64 --name lysto64_pix2pix \
#     --dataroot /home/sszabados/datasets/lysto64_random_crop_pix2pix/AB/ \
#     --direction AtoB --batch_size 32 --load_size 64 --crop_size 64 --preprocess none \
#     --n_epochs 161 --save_epoch_freq 5 --display_freq 1000 --print_freq 1000 --display_winsize 64


# Train pix2pix on PET-CT
# -----------------------------------------------
python train.py --model pix2pix --netG unet_256 --name PET-CT_pix2pix \
    --dataroot /home/sszabados/datasets/ct_pet/AB/ \
    --direction AtoB --batch_size 32 --load_size 256 --crop_size 256 --preprocess none \
    --n_epochs 161 --save_epoch_freq 10 --display_freq 1000 --print_freq 1000 --display_winsize 256
