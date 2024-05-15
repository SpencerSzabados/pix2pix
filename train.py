"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

import os
import pathlib as path
import numpy as np
import torch as th
from evaluations.fid_score import calculate_fid_given_paths
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim_measure



if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    # Evaluation metrics 
    MAE = []
    MSE = []
    fids = []
    SSIM = []

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0: # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

            with th.no_grad():
                print("Computing FID of model and L1 score")
                # sample 50,000 images for computeing fid against train set
                print("\nGenerating fid samples...", flush=True)
            
                sampling_dataset = create_dataset(opt)
                num_batches = 50000//len(sampling_dataset)
                for _ in range(num_batches):
                    for i, data in enumerate(sampling_dataset):
                        model.set_input(data)
                        model.forward()
                        sample = model.fake_B.cpu().detach()
                        sample = sample.permute(0,2,3,1).numpy()
                        sample = (sample+1)/2. 
                        for j in range(len(sample)):
                            plt.imsave('/home/sszabados/checkpoints/pix2pix/ct_pet/fid_samples/image_{}.png'.format(i*opt.batch_size+j), sample[j]) # When generating multichannel data
                            # plt.imsave('fid/{}/image_{}.JPEG'.format("fid_samples", i*batch_size+j), samples[j,:,:,0], cmap='gray') # When generating single channel data
                print("\nfinished generating fid samples.", flush=True)

                print("\ncomputing fid...")
                dir2ref = "/home/sszabados/datasets/ct_pet/B/pet/"
                dir2gen = "/home/sszabados/checkpoints/pix2pix/ct_pet/fid_samples/"
                fid_value = 0
                try:
                    fid_value = calculate_fid_given_paths(
                        paths = [dir2ref, dir2gen],
                        batch_size = 128,
                        device = "cuda:1",
                        img_size = 256,
                        dims = 2048,
                        num_workers = 1,
                        eqv = 'H' 
                    )
                except ValueError:
                    fid_value = np.inf
                fids.append([epoch,fid_value])
                # Incrementally save fids after each epoch
                os.makedirs('/home/sszabados/checkpoints/pix2pix/ct_pet/metrics/', exist_ok=True)
                np.save('/home/sszabados/checkpoints/pix2pix/ct_pet/metrics/fid.npy', np.array(fids))
                print(f"FID: {fid_value}")

                print("Computing MSE, MSA, SSIM, loss...")
                val_opt = opt
                val_opt.phase = "val"
                sampling_dataset = create_dataset(val_opt)
                print(len(sampling_dataset))

                ref_samples = []
                samples = []
                for i, data in enumerate(sampling_dataset):
                    model.set_input(data)
                    model.forward()
                    sample = model.fake_B.cpu().detach()
                    samples.append(sample)
                    ref_samples.append(model.real_B.cpu().detach())
                samples = th.cat(samples, dim=0)
                ref_samples = th.cat(ref_samples, dim=0)

                mse =  F.mse_loss(ref_samples, samples).item()
                mae = F.l1_loss(ref_samples, samples).item()

                ssim_T = ssim_measure(data_range=(-1,1))
                # ssim_score = ssim_T(preds=samples, target=ref_samples).item()
                samples_len = len(samples)
                num_batches = samples_len//100
                for i in range(0, num_batches):
                    s = i*100
                    e = min(s+100, samples_len)
                    ssim_T.update(samples[s:e], ref_samples[s:e])
                ssim_score = ssim_T.compute().item()
                
                MSE.append([epoch, mse])
                MAE.append([epoch, mae])
                SSIM.append([epoch, ssim_score])
                os.makedirs('/home/sszabados/checkpoints/pix2pix/ct_pet/metrics/', exist_ok=True)
                np.save('/home/sszabados/checkpoints/pix2pix/ct_pet/metrics/mse.npy', np.array(MSE))
                os.makedirs('/home/sszabados/checkpoints/pix2pix/ct_pet/metrics/', exist_ok=True)
                np.save('/home/sszabados/checkpoints/pix2pix/ct_pet/metrics/mae.npy', np.array(MAE))
                os.makedirs('/home/sszabados/checkpoints/pix2pix/ct_pet/metrics/', exist_ok=True)
                np.save('/home/sszabados/checkpoints/pix2pix/ct_pet/metrics/ssim.npy', np.array(SSIM))

                print(f"MSE: {mse}, MAE: {mae}, SSIM: {ssim_score}")


        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
