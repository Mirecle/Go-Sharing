import numpy as np
import os
import re
import csv
import time
import pickle
import logging

import torch
from torchvision import datasets, transforms
import torchvision.utils
from torch.utils import data
import torch.nn.functional as F

from options import HiDDenConfiguration, TrainingOptions
from model.hidden import Hidden


def image_to_tensor(image):
    """
    Transforms a numpy-image into torch tensor
    :param image: (batch_size x height x width x channels) uint8 array
    :return: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    """
    image_tensor = torch.Tensor(image)
    image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.permute(0, 3, 1, 2)
    image_tensor = image_tensor / 127.5 - 1
    return image_tensor


def tensor_to_image(tensor):
    """
    Transforms a torch tensor into numpy uint8 array (image)
    :param tensor: (batch_size x channels x height x width) torch tensor in range [-1.0, 1.0]
    :return: (batch_size x height x width x channels) uint8 array
    """
    image = tensor.permute(0, 2, 3, 1).cpu().numpy()
    image = (image + 1) * 127.5
    return np.clip(image, 0, 255).astype(np.uint8)


def save_images(original_images, watermarked_images,noised_images, epoch, folder, resize_to=None):
    images = original_images[:original_images.shape[0], :, :, :].cpu()
    watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()
    noised_images=noised_images[:noised_images.shape[0], :, :, :].cpu()
    # scale values to range [0, 1] from original range of [-1, 1]
    images = (images + 1) / 2
    watermarked_images = (watermarked_images + 1) / 2
    noised_images=(noised_images+1)/2
    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
        watermarked_images = F.interpolate(watermarked_images, size=resize_to)
        noised_images=F.interpolate(noised_images,size=resize_to)

        filename_original_png= os.path.join(folder, 'original{}.png'.format(epoch))
        torchvision.utils.save_image( images, filename_original_png, normalize=False)

        filename_watermarked_png = os.path.join(folder, 'watermarked{}.png'.format(epoch))
        torchvision.utils.save_image(watermarked_images, filename_watermarked_png, normalize=False)

        noised_png = os.path.join(folder, 'noised{}.png'.format(epoch))
        torchvision.utils.save_image(noised_images, noised_png, normalize=False)



def mse(org_img, img):
    # the way to calculate gray image and colorful image is different
    list=[0,0,0,0]
    for i in range(len(img)):


        if len(img[1]) == 3:
            diff_b = org_img[1][:][:][0] - img[1][:][:][0]
            diff_g = org_img[1][:][:][1] - img[1][:][:][1]
            diff_r = org_img[1][:][:][2] - img[1][:][:][2]
            diff_b = diff_b.flatten()
            diff_g = diff_g.flatten()
            diff_r = diff_r.flatten()
            list[0]=list[0]+ np.mean(diff_b**2)/100
            list[1] = list[1]+np.mean(diff_g**2)/100
            list[2]= list[2]+np.mean(diff_r**2)/100
            list[3]= list[3]+ np.mean(diff_b ** 2 + diff_g ** 2 + diff_r ** 2)/300
        else:
            diff = org_img[i] - img[i]
            diff = diff.flatten()
            result = np.mean(diff ** 2)
            list[3]=list[3]+result

    return list
def psnr(org_img, img):
    if len(mse(org_img, img))!=1:
        return 10*np.log10(255**2/mse(org_img, img)[0]),10*np.log10(255**2/mse(org_img, img)[1]),10*np.log10(255**2/mse(org_img, img)[2]),10*np.log10(255**2/mse(org_img, img)[3])
    else:
        return 10*np.log10(255**2/mse(org_img, img)[0])
def transfer(images):
    # images = (images + 1) / 2
    batch_encoded_image = images.cpu().detach().numpy() * 255
    # batch_encoded_image = batch_encoded_image.transpose((0, 2, 3, 1))
    return batch_encoded_image
def SPNR_save(original_images, watermarked_images, resize_to=None):
    images = original_images
    watermarked_images = watermarked_images

    # scale values to range [0, 1] from original range of [-1, 1]
    images = (images + 1) / 2
    watermarked_images = (watermarked_images + 1) / 2


    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
        watermarked_images = F.interpolate(watermarked_images, size=resize_to)

    images=transfer(images)
    watermarked_images=transfer(watermarked_images)

    return psnr(images,watermarked_images)


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def last_checkpoint_from_folder(folder: str):
    last_file = sorted_nicely(os.listdir(folder))[-1]
    last_file = os.path.join(folder, last_file)
    return last_file


def save_checkpoint(model: Hidden, experiment_name: str, epoch: int, checkpoint_folder: str):
    """ Saves a checkpoint at the end of an epoch. """
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint_filename = f'{experiment_name}--epoch-{epoch}.pyt'
    checkpoint_filename = os.path.join(checkpoint_folder, checkpoint_filename)
    logging.info('Saving checkpoint to {}'.format(checkpoint_filename))
    checkpoint = {
        'enc-model': model.encoder.state_dict(),
        'dec-model': model.decoder.state_dict(),
        'enc-dec-optim': model.optimizer_enc_dec.state_dict(),
        'discrim-model': model.discriminator.state_dict(),
        'discrim-optim': model.optimizer_discrim.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_filename)
    logging.info('Saving checkpoint done.')
def save_checkpoint_decode(model: Hidden, experiment_name: str, epoch: int, checkpoint_folder: str):
    """ Saves a checkpoint at the end of an epoch. """
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint_filename = f'{experiment_name}--epoch-{epoch}.pyt'
    checkpoint_filename = os.path.join(checkpoint_folder, checkpoint_filename)
    logging.info('Saving checkpoint to {}'.format(checkpoint_filename))
    checkpoint = {
        'enc-model': model.encoder.state_dict(),
        'dec-model': model.decoder.state_dict(),
        'enc-dec-optim': model.optimizer_enc_dec.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_filename)
    logging.info('Saving checkpoint done.')

# def load_checkpoint(hidden_net: Hidden, options: Options, this_run_folder: str):
def load_last_checkpoint(checkpoint_folder):
    """ Load the last checkpoint from the given folder """
    last_checkpoint_file = last_checkpoint_from_folder(checkpoint_folder)
    checkpoint = torch.load(last_checkpoint_file)

    return checkpoint, last_checkpoint_file


def model_from_checkpoint(hidden_net, checkpoint):

    """ Restores the hidden_net object from a checkpoint object """
    encoder={}
    decoder={}

    #hidden_net.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'])
    hidden_net.encoder.load_state_dict(checkpoint['enc-model'])
    hidden_net.decoder.load_state_dict(checkpoint['dec-model'])
    hidden_net.optimizer_enc_dec.load_state_dict(checkpoint['enc-dec-optim'])
    hidden_net.discriminator.load_state_dict(checkpoint['discrim-model'])
    hidden_net.optimizer_discrim.load_state_dict(checkpoint['discrim-optim'])
    # try:
    #     hidden_net.optimizer_enc_dec.load_state_dict(checkpoint['enc-dec-optim'])
    # except Exception as e:
       # print('Can not load optimizer_enc_dec')
    #hidden_net.discriminator.load_state_dict(checkpoint['discrim-model'])
   # hidden_net.optimizer_discrim.load_state_dict(checkpoint['discrim-optim'])
def model_from_checkpoint_only_decode(hidden_net, checkpoint):

    """ Restores the hidden_net object from a checkpoint object """
    encoder={}
    decoder={}

    #hidden_net.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'])
    hidden_net.encoder.load_state_dict(checkpoint['enc-model'])
    hidden_net.decoder.load_state_dict(checkpoint['dec-model'])


def load_options(options_file_name) -> (TrainingOptions, HiDDenConfiguration, dict):
    """ Loads the training, model, and noise configurations from the given folder """
    with open(os.path.join(options_file_name), 'rb') as f:
        train_options = pickle.load(f)
        noise_config = pickle.load(f)
        hidden_config = pickle.load(f)
        # for backward-capability. Some models were trained and saved before .enable_fp16 was added
        if not hasattr(hidden_config, 'enable_fp16'):
            setattr(hidden_config, 'enable_fp16', False)

    return train_options, hidden_config, noise_config


def get_data_loaders(hidden_config: HiDDenConfiguration, train_options: TrainingOptions):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((hidden_config.H, hidden_config.W), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'test': transforms.Compose([
            transforms.CenterCrop((hidden_config.H, hidden_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }

    train_images = datasets.ImageFolder(train_options.train_folder, data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=train_options.batch_size, shuffle=True,
                                               num_workers=4)

    validation_images = datasets.ImageFolder(train_options.validation_folder, data_transforms['test'])
    validation_loader = torch.utils.data.DataLoader(validation_images, batch_size=train_options.batch_size,
                                                    shuffle=False, num_workers=4)

    return train_loader, validation_loader



def get_test_data_loaders(hidden_config: HiDDenConfiguration, train_options: TrainingOptions):
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    data_transforms = {

        'test': transforms.Compose([
            transforms.CenterCrop((hidden_config.H, hidden_config.W)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }


    validation_images = datasets.ImageFolder(train_options.validation_folder, data_transforms['test'])
    validation_loader = torch.utils.data.DataLoader(validation_images, batch_size=train_options.batch_size,
                                                    shuffle=False, num_workers=4)

    return validation_loader

def log_progress(losses_accu):
    log_print_helper(losses_accu, logging.info)


def print_progress(losses_accu):
    log_print_helper(losses_accu, print)


def log_print_helper(losses_accu, log_or_print_func):
    max_len = max([len(loss_name) for loss_name in losses_accu])
    for loss_name, loss_value in losses_accu.items():
        log_or_print_func(loss_name.ljust(max_len + 4) + '{:.4f}'.format(loss_value.avg))


def create_folder_for_run(runs_folder, experiment_name):
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)

    this_run_folder = os.path.join(runs_folder, f'{experiment_name} {time.strftime("%Y.%m.%d--%H-%M-%S")}')

    os.makedirs(this_run_folder)
    os.makedirs(os.path.join(this_run_folder, 'checkpoints'))
    os.makedirs(os.path.join(this_run_folder, 'images'))

    return this_run_folder


def write_losses(file_name, losses_accu, epoch, duration):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()] + ['duration']
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.4f}'.format(loss_avg.avg) for loss_avg in losses_accu.values()] + [
            '{:.0f}'.format(duration)]
        writer.writerow(row_to_write)