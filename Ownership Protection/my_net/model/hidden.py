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
        self.discriminator = Discriminator(configuration).to(device)
        self.optimizer_enc_dec = torch.optim.Adam(itertools.chain(self.encoder.parameters(), self.decoder.parameters()))
        self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters())

        if configuration.use_vgg:
            self.vgg_loss = VGGLoss(3, 1, False)
            self.vgg_loss.to(device)
        else:
            self.vgg_loss = None

        self.config = configuration
        self.device = device

        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)

        # Defined the labels used for training the discriminator/adversarial loss
        self.cover_label = 1
        self.encoded_label = 0

        self.tb_logger = tb_logger
        # if tb_logger is not None:
        #     from tensorboard_logger import TensorBoardLogger
        #     encoder_final = self.encoder_decoder.encoder._modules['final_layer']
        #     encoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/encoder_out'))
        #     decoder_final = self.encoder_decoder.decoder._modules['linear']
        #     decoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/decoder_out'))
        #     discrim_final = self.discriminator._modules['linear']
        #     discrim_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/discrim_out'))


    def train_on_batch(self, batch: list,only_decode):
        """
        Trains the network on a single batch consisting of images and messages
        :param batch: batch of training data, in the form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        images, messages = batch

        batch_size = images.shape[0]
        self.encoder.train()# 训练加水印解水印
        self.decoder.train()
        self.discriminator.train() #鉴别器
        with torch.enable_grad():
            # ---------------- Train the discriminator -----------------------------
            self.optimizer_discrim.zero_grad()
            # train on cover
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

            d_on_cover = self.discriminator(images)         #d_on_cover是图片进入鉴别器后的内容
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover.float())    #d_on_cover 和d_target_label_cover的比较后得到的损失
            d_loss_on_cover.backward()                        #损失返向传播

            # train on fake
            encoded_image = self.encoder(images, messages, 0.2)
            #noised_and_cover, info = randomNoise(encoded_image, 'low')
            noised_image = encoded_image
            # noised_image=encoded_image
            decoded_message = self.decoder(noised_image)
            d_on_encoded = self.discriminator(encoded_image.detach())                     #和73行对应，73行是原图进入鉴别器后的内容，本行是水印后图片进入鉴别器后的损失
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded.float())  #计算损失

            d_loss_on_encoded.backward()
            self.optimizer_discrim.step()

            # --------------Train the generator (encoder-decoder) ---------------------
            self.optimizer_enc_dec.zero_grad()
            # target label for encoded images should be 'cover', because we want to fool the discriminator
            d_on_encoded_for_enc = self.discriminator(encoded_image)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded.float()) #计算加密后图片被鉴别器识别的损失

            if self.vgg_loss == None:
                g_loss_enc = self.mse_loss(encoded_image, images)
            else:
                vgg_on_cov = self.vgg_loss(images)
                vgg_on_enc = self.vgg_loss(encoded_image)
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

            g_loss_dec = self.mse_loss(decoded_message, messages) #解密后massage 的损失
            g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
                     + self.config.decoder_loss * g_loss_dec  #总损失

            g_loss.backward() # 损失反向传播
            self.optimizer_enc_dec.step()

        decoded_rounded = decoded_message.detach().cpu().numpy().round().clip(0, 1) #得到具体的解码massage 内容
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1]) #具体的平均比特错误

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item()
        }
        return losses, (encoded_image, noised_image, decoded_message)

    def validate_on_batch(self, batch: list,only_decode):
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

        self.decoder.eval()
        self.encoder.eval()
        self.discriminator.eval()
        with torch.no_grad():
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)#cover_label=1
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)#encoded_label=0
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover.float())#计算得到的d_on_cover和全1的损失（把原图识别为原图的损失）

            encoded_image = self.encoder(images, messages, 0.2)
            #noised_and_cover, info = randomNoise(encoded_image, 'low')
            noised_image = encoded_image
            # noised_image=encoded_image
            decoded_message = self.decoder(noised_image)
            d_on_encoded = self.discriminator(encoded_image.detach())  # 和73行对应，73行是原图进入鉴别器后的内容，本行是水印后图片进入鉴别器后的损失
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded.float())  # 计算损失

            d_on_encoded = self.discriminator(encoded_image)
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded.float())#计算得到的d_loss_on_encoded 和全0的损失（把encode 图片识别为生成图的损失）

            d_on_encoded_for_enc = self.discriminator(encoded_image)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded.float())#计算得到的d_loss_on_encoded 和全1的损失，（把encode 图片识别为原图的损失）

            if self.vgg_loss is None:
                g_loss_enc = self.mse_loss(encoded_image, images) #生成图片和原图之间的损失
            else:
                vgg_on_cov = self.vgg_loss(images)
                vgg_on_enc = self.vgg_loss(encoded_image)
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

            g_loss_dec = self.mse_loss(decoded_message, messages)
            g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
                     + self.config.decoder_loss * g_loss_dec#config.adversarial_loss=0.001，encoder_loss=0.7,decoder_loss=1

        decoded_rounded = decoded_message.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(), #越大，生成图和原图越不像 #越大鉴别器越好
            'dec_mse        ': g_loss_dec.item(),#越大，massage错误率越大
            'bitwise-error  ': bitwise_avg_err, #越大，比特错误率越高
            'adversarial_bce': g_loss_adv.item(),#越大，鉴别器把生成图片识别为原图的概率越低   #越大鉴别器越好
            'discr_cover_bce': d_loss_on_cover.item(), #越小，鉴别器的效果越好（越不容易把原图错判为生成图）#越小鉴别器越好
            'discr_encod_bce': d_loss_on_encoded.item()# 越小，鉴别器把生成图片识别为生成图片的能力越好 #越小鉴别器越好
        }
        return losses, (encoded_image, noised_image, decoded_message)

    def to_stirng(self):
        return '{}\n{}\n{}'.format(str(self.encoder),str(self.decoder), str(self.discriminator))
