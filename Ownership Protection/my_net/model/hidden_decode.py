import numpy as np
import torch
import torch.nn as nn

from options import HiDDenConfiguration
from model.discriminator import Discriminator
from model.encoder_decoder import EncoderDecoder
from vgg_loss import VGGLoss

from model.two_encoder import Encoder
from model.two_decoder import Decoder
from Noise import randomNoise
import time
import itertools
class Hidden:
    def __init__(self, configuration: HiDDenConfiguration, device: torch.device, tb_logger):
        """
        :param configuration: Configuration for the net, such as the size of the input image, number of channels in the intermediate layers, etc.
        :param device: torch.device object, CPU or GPU
        :param noiser: Object representing stacked noise layers.
        :param tb_logger: Optional TensorboardX logger object, if specified -- enables Tensorboard logging
        """
        super(Hidden, self).__init__()
        self.encoder=Encoder(3,30).to(device)
        self.decoder = Decoder(3, 30).to(device)

        self.optimizer_enc_dec = torch.optim.Adam(self.decoder.parameters(), lr=1e-5)




        self.config = configuration
        self.device = device


        self.mse_loss = nn.MSELoss().to(device)

        # Defined the labels used for training the discriminator/adversarial loss
        self.cover_label = 1
        self.encoded_label = 0


    def train_on_batch(self, batch: list):
        """
        Trains the network on a single batch consisting of images and messages
        :param batch: batch of training data, in the form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        images, messages = batch

        batch_size = images.shape[0]
        self.encoder.train()
        self.decoder.train()
        with torch.enable_grad():
            # --------------Train the generator (encoder-decoder) ---------------------
            self.optimizer_enc_dec.zero_grad()
            encoded_image = self.encoder(images, messages, 0.2)
            noised_and_cover, info = randomNoise(encoded_image, 'low')

            noised_image=noised_and_cover
            decoded_message = self.decoder(noised_image)
            # target label for encoded images should be 'cover', because we want to fool the discriminator



            g_loss_dec = self.mse_loss(decoded_message, messages) #解密后massage 的损失

            g_loss_dec.backward() # 损失反向传播
            self.optimizer_enc_dec.step()

        decoded_rounded = decoded_message.detach().cpu().numpy().round().clip(0, 1) #得到具体的解码massage 内容
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1]) #具体的平均比特错误

        losses = {
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,
        }
        return losses, (encoded_image, noised_image, decoded_message)

    def validate_on_batch(self, batch: list):
        """
        Runs validation on a single batch of data consisting of images and messages
        :param batch: batch of validation data, in form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        # if TensorboardX logging is enabled, save some of the tensors.
        # if self.tb_logger is not None:
        #     encoder_final = self.encoder_decoder.encoder._modules['final_layer']
        #     self.tb_logger.add_tensor('weights/encoder_out', encoder_final.weight)
        #     decoder_final = self.encoder_decoder.decoder._modules['linear']
        #     self.tb_logger.add_tensor('weights/decoder_out', decoder_final.weight)
        #     discrim_final = self.discriminator._modules['linear']
        #     self.tb_logger.add_tensor('weights/discrim_out', discrim_final.weight)

        images, messages = batch

        batch_size = images.shape[0]
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            starttime = time.time()
            encoded_image = self.encoder(images, messages, 0.2)
            endtime = time.time()
            result = endtime - starttime
            print(result)
            noise_photo=encoded_image.detach().clone()
            noised_and_cover, info = randomNoise(noise_photo, 'low')

            noised_image=noised_and_cover


            decoded_message = self.decoder(noised_image)


            g_loss_dec = self.mse_loss(decoded_message, messages)

        decoded_rounded = decoded_message.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {

            'dec_mse        ': g_loss_dec.item(),#越大，massage错误率越大
            'bitwise-error  ': bitwise_avg_err, #越大，比特错误率越高
        }
        return losses, (encoded_image, noised_image, decoded_message)

    def to_stirng(self):
        return '{}\n{}'.format(str(self.encoder),str(self.decoder))
