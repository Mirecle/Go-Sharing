import torch.nn as nn
from model.two_encoder import Encoder
from model.two_decoder import Decoder
from options import HiDDenConfiguration

from Noise import randomNoise

class EncoderDecoder(nn.Module):
    """
    Combines Encoder->Noiser->Decoder into single pipeline.
    The input is the cover image and the watermark message. The module inserts the watermark into the image
    (obtaining encoded_image), then applies Noise layers (obtaining noised_image), then passes the noised_image
    to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
    a three-tuple: (encoded_image, noised_image, decoded_message)
    """
    def __init__(self, config: HiDDenConfiguration,onlydecode):

        super(EncoderDecoder, self).__init__()
        #self.encoder = Encoder(config,onlydecode)   老程序代码
        self.encoder = Encoder(3, 30)

        for p in self.parameters():
            p.requires_grad = False

        #self.decoder = Decoder(config)
        self.decoder = Decoder(3,30)
    def forward(self, image, message):
        encoded_image = self.encoder(image, message,0.2)
        noised_and_cover,info = randomNoise(encoded_image,'low')
        noised_image = noised_and_cover
        #noised_image=encoded_image
        decoded_message = self.decoder(noised_image)
        return encoded_image, noised_image, decoded_message
