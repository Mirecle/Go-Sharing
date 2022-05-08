import os
import pprint
import argparse
import torch
import pickle
import utils
import logging
import sys

from options import *
from model.hidden import Hidden
from noise_layers.noiser import Noiser
from noise_argparser import NoiseArgParser

from train import train


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



    checkpoint = None
    loaded_checkpoint_file_name = None

    command='new'
    folder='.\\runs\\two_stage 2022.01.06--17-48-23\\'
    priv_folder='.\\runs\\fixencoder 2022.01.05--00-58-36\\'
    data_dir='..\\HiDDeN-master\\data\\'
    epochs=300


    size=128
    noise=None
    message=30
    enable_fp16=False
    tensorboard=False
    batch_size=12
    name='first_no_noise'

    if command == 'continue':
        this_run_folder = priv_folder
        options_file = os.path.join(this_run_folder, 'options-and-config.pickle')
        train_options, hidden_config, noise_config = utils.load_options(options_file)
        checkpoint, loaded_checkpoint_file_name = utils.load_last_checkpoint(os.path.join(this_run_folder, 'checkpoints'))
        model = Hidden(hidden_config, device, None)
        utils.model_from_checkpoint(model, checkpoint)
        train_options.start_epoch = checkpoint['epoch'] + 1
        if data_dir is not None:
            train_options.train_folder = os.path.join(data_dir, 'train')
            train_options.validation_folder = os.path.join(data_dir, 'val')
        if epochs is not None:
            if train_options.start_epoch < epochs:
                train_options.number_of_epochs = epochs
            else:
                print(f'Command-line specifies of number of epochs = {epochs}, but folder={folder} '
                      f'already contains checkpoint for epoch = {train_options.start_epoch}.')
                exit(1)

    else:
        assert command == 'new'
        start_epoch = 1



        train_options = TrainingOptions(
            batch_size=batch_size,
            number_of_epochs=epochs,
            train_folder=os.path.join(data_dir, 'train'),
            validation_folder=os.path.join(data_dir, 'val'),
            runs_folder=os.path.join('.', 'runs'),
            start_epoch=start_epoch,
            experiment_name=name)

        noise_config = noise if noise is not None else []
        hidden_config = HiDDenConfiguration(H=size, W=size,
                                            message_length=message,
                                            encoder_blocks=4, encoder_channels=64,
                                            decoder_blocks=7, decoder_channels=64,
                                            use_discriminator=True,
                                            use_vgg=False,
                                            discriminator_blocks=3, discriminator_channels=64,
                                            decoder_loss=1,
                                            encoder_loss=0.7,
                                            adversarial_loss=1e-3,
                                            enable_fp16=enable_fp16
                                            )

        this_run_folder = utils.create_folder_for_run(train_options.runs_folder, name)
        with open(os.path.join(this_run_folder, 'options-and-config.pickle'), 'wb+') as f:
            pickle.dump(train_options, f)
            pickle.dump(noise_config, f)
            pickle.dump(hidden_config, f)

        ############################载入模型#########################################################
        #checkpoint, loaded_checkpoint_file_name = utils.load_last_checkpoint(os.path.join(folder, 'checkpoints'))

        model = Hidden(hidden_config, device, None)
        #utils.model_from_checkpoint(model, checkpoint)

    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(this_run_folder, f'{train_options.experiment_name}.log')),
                            logging.StreamHandler(sys.stdout)
                        ])



    if command == 'continue':
        # if we are continuing, we have to load the model params
        assert checkpoint is not None
        logging.info(f'Loading checkpoint from file {loaded_checkpoint_file_name}')
        utils.model_from_checkpoint(model, checkpoint)

    logging.info('HiDDeN model: {}\n'.format(model.to_stirng()))
    logging.info('Model Configuration:\n')
    logging.info(pprint.pformat(vars(hidden_config)))
    logging.info('\nNoise configuration:\n')
    logging.info(pprint.pformat(str(noise_config)))
    logging.info('\nTraining train_options:\n')
    logging.info(pprint.pformat(vars(train_options)))

    train(model, True,device, hidden_config, train_options, this_run_folder, None)


if __name__ == '__main__':
    main()
