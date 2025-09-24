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
import sys
import os
import logging
from datetime import datetime
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import init_ddp, cleanup_ddp


def setup_logging():
    """Setup comprehensive logging system"""
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/train_log_{timestamp}.txt"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Log the command that started training
    command_line = ' '.join(sys.argv)
    logging.info(f"=== TRAINING STARTED ===")
    logging.info(f"Command: python {command_line}")
    logging.info(f"Log file: {log_filename}")
    logging.info(f"Working directory: {os.getcwd()}")
    
    return log_filename

class LoggingVisualizer(Visualizer):
    """Extended visualizer with logging capabilities"""
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """Print and log current losses"""
        # Call parent method for normal display
        super().print_current_losses(epoch, iters, losses, t_comp, t_data)
        
        # Log to file
        loss_str = ', '.join([f'{k}: {v:.3f}' for k, v in losses.items()])
        logging.info(f"[Rank {self.opt.gpu_ids[0] if self.opt.gpu_ids else 0}] (epoch: {epoch}, iters: {iters}, time: {t_comp:.3f}, data: {t_data:.3f}) , {loss_str}")

if __name__ == "__main__":
    try:
        # Setup logging first
        log_file = setup_logging()
        
        opt = TrainOptions().parse()  # get training options
        logging.info(f"Training options parsed successfully")
        
        opt.device = init_ddp()
        logging.info(f"Device initialized: {opt.device}")
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
        dataset_size = len(dataset)  # get the number of images in the dataset.
        print(f"The number of training images = {dataset_size}")
        logging.info(f"Dataset created: {opt.dataset_mode}, Size: {dataset_size}")

        model = create_model(opt)  # create a model given opt.model and other options
        model.setup(opt)  # regular setup: load and print networks; create schedulers
        logging.info(f"Model created: {opt.model}")
        
        visualizer = LoggingVisualizer(opt)  # create a visualizer that display/save images and plots
        logging.info(f"Visualizer initialized")
        
        total_iters = 0  # the total number of training iterations
        logging.info(f"Starting training: epochs {opt.epoch_count} to {opt.n_epochs + opt.n_epochs_decay}")
        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()  # timer for data loading per iteration
            epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
            visualizer.reset()
            logging.info(f"=== EPOCH {epoch} STARTED ===")
            
            # Set epoch for DistributedSampler
            if hasattr(dataset, "set_epoch"):
                dataset.set_epoch(epoch)

            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()  # timer for computation per iteration
                if total_iters % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data)  # unpack data from dataset and apply preprocessing
                model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

                if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                    save_result = total_iters % opt.update_html_freq == 0
                    model.compute_visuals()
                    visualizer.display_current_results(model.get_current_visuals(), epoch, total_iters, save_result)

                if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                    visualizer.plot_current_losses(total_iters, losses)

                if total_iters % opt.save_latest_freq == 0:  # cache our latest model every <save_latest_freq> iterations
                    print(f"saving the latest model (epoch {epoch}, total_iters {total_iters})")
                    logging.info(f"Saving latest model (epoch {epoch}, total_iters {total_iters})")
                    save_suffix = f"iter_{total_iters}" if opt.save_by_iter else "latest"
                    model.save_networks(save_suffix)

                iter_data_time = time.time()

            model.update_learning_rate()  # update learning rates at the end of every epoch
            current_lr = model.optimizers[0].param_groups[0]['lr']
            logging.info(f"Learning rate updated to: {current_lr:.6f}")

            if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                print(f"saving the model at the end of epoch {epoch}, iters {total_iters}")
                logging.info(f"Saving model at end of epoch {epoch}, iters {total_iters}")
                model.save_networks("latest")
                model.save_networks(epoch)

            epoch_time = time.time() - epoch_start_time
            print(f"End of epoch {epoch} / {opt.n_epochs + opt.n_epochs_decay} \t Time Taken: {epoch_time:.0f} sec")
            logging.info(f"=== EPOCH {epoch} COMPLETED === Time: {epoch_time:.0f}s")

        logging.info(f"=== TRAINING COMPLETED SUCCESSFULLY ===")
        cleanup_ddp()
        
    except Exception as e:
        logging.error(f"=== TRAINING FAILED ===")
        logging.error(f"Error: {str(e)}")
        logging.error(f"Error type: {type(e).__name__}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise e
    finally:
        logging.info(f"=== TRAINING SESSION ENDED ===")
