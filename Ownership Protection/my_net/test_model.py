import random

import torch
import torch.nn
import argparse
import os
import numpy as np
from options import HiDDenConfiguration
import main_for_decode
import utils
from model.hidden_decode import *
import time
from PIL import Image
import torchvision.transforms.functional as TF
import train_decode
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        if val != np.nan and val != np.inf:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
def randomCrop(img, height, width):
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    img = img[y:y+height, x:x+width]
    return img


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    data_dir='.\\data\\'
    epochs=600

    images_to_save = 8
    saved_images_size = (512, 512)
    size=128
    noise=None
    message=30
    enable_fp16=False
    tensorboard=False
    batch_size=100
    train_options = main_for_decode.TrainingOptions(
        batch_size=batch_size,
        number_of_epochs=epochs,
        train_folder=os.path.join(data_dir, 'train'),
        validation_folder=os.path.join(data_dir, 'val'),
        runs_folder=os.path.join('.', 'runs'),
        start_epoch=1,
        experiment_name='sda'
        )
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
    checkpoint_file=r".\runs\decode_test 2022.01.17--21-41-04\checkpoints\decode_test--epoch-458.pyt"
    #checkpoint_file = r"D:\mechine_learning\HiDDeN-master\experiments\dropout-0.55-0.6\checkpoints\epoch-300.pyt"
    #options_file=r".\runs\decode_test 2022.01.17--21-41-04\options-and-config.pickle"

    val_data = utils.get_test_data_loaders(hidden_config, train_options)
    #train_options, hidden_config, noise_config = utils.load_options(options_file)
    #noiser = Noiser(noise_config,device)

    checkpoint = torch.load(checkpoint_file)
    hidden_net = Hidden(hidden_config, device, None)
    #utils.model_from_checkpoint(hidden_net, checkpoint)
    utils.model_from_checkpoint_only_decode(hidden_net, checkpoint)
    first_iteration = True
    validation_losses = train_decode.defaultdict(AverageMeter)
    sums=[0,0,0,0]
    for image, _ in val_data:
        image = image.to(device)
        message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], hidden_config.message_length))).to(device)

        losses, (encoded_images, noised_images, decoded_messages) = hidden_net.validate_on_batch([image, message])

        for name, loss in losses.items():
            validation_losses[name].update(loss)
        if first_iteration:
            if hidden_config.enable_fp16:
                image = image.float()
                encoded_images = encoded_images.float()
                noised_images = noised_images.float()
            randnum=random.randint(1,98)
            utils.save_images(image.cpu()[randnum:randnum+1, :, :, :],
                              encoded_images[randnum:randnum+1, :, :, :].cpu(),
                              noised_images[randnum:randnum+1, :, :, :].cpu(),
                              0,
                              os.path.join(r'.\test_result', 'None'), resize_to=saved_images_size)
            first_iteration = False
        SPNR=(utils.SPNR_save(image.cpu(), encoded_images.cpu()))
        #sum+=SPNR
        for i in range(len(SPNR)):
            sums[i] += SPNR[i] / 10
        print(SPNR)
    print("平均：{}".format(sums))

    print([loss_name.strip() for loss_name in validation_losses.keys()])
    print(['{:.4f}'.format(loss_avg.avg) for loss_avg in validation_losses.values()])
    # image_pil = Image.open(source_image)
    # image = randomCrop(np.array(image_pil), hidden_config.H, hidden_config.W)
    # image_tensor = TF.to_tensor(image).to(device)
    # image_tensor = image_tensor * 2 - 1  # transform from [0, 1] to [-1, 1]
    # image_tensor.unsqueeze_(0)
    #
    # # for t in range(args.times):
    # message = torch.Tensor(np.random.choice([0, 1], (image_tensor.shape[0],
    #                                                 hidden_config.message_length))).to(device)
    # losses, (encoded_images, noised_images, decoded_messages) = hidden_net.validate_on_batch([image_tensor, message])
    # decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
    # message_detached = message.detach().cpu().numpy()
    # print('original: {}'.format(message_detached))
    # print('decoded : {}'.format(decoded_rounded))
    # print('losses: {}'.format(losses))
    # print('error : {:.3f}'.format(np.mean(np.abs(decoded_rounded - message_detached))))
    #utils.save_images(image_tensor.cpu(), encoded_images.cpu(),noised_images.cpu(), 'test', '.', resize_to=(256, 256))#是否需要图片保存

    # bitwise_avg_err = np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy()))/(image_tensor.shape[0] * messages.shape[1])



if __name__ == '__main__':
    main()
