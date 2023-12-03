# python 3.7
''''''
import cv2
import numpy as np
import sys

from models.stylegan_generator import StyleGANGenerator
from models.stylegan_encoder import StyleGANEncoder
from models.perceptual_model import PerceptualModel


class BaseStyleGANInverter(object):

    def __init__(self, model_name, logger=None):
        self.logger = logger
        self.model_name = model_name
        self.gan_type = 'stylegan'

        self.G = StyleGANGenerator(self.model_name, self.logger)
        self.E = StyleGANEncoder(self.model_name, self.logger)
        self.F = PerceptualModel(min_val=self.G.min_val, max_val=self.G.max_val)
        self.encode_dim = [self.G.num_layers, self.G.w_space_dim]
        self.run_device = self.G.run_device
        assert list(self.encode_dim) == list(self.E.encode_dim)

        assert self.G.gan_type == self.gan_type
        assert self.E.gan_type == self.gan_type

    def preprocess(self, image):
        """Preprocesses a single image.

    This function assumes the input numpy array is with shape [height, width,
    channel], channel order `RGB`, and pixel range [0, 255].

    The returned image is with shape [channel, new_height, new_width], where
    `new_height` and `new_width` are specified by the given generative model.
    The channel order of returned image is also specified by the generative
    model. The pixel range is shifted to [min_val, max_val], where `min_val` and
    `max_val` are also specified by the generative model.
    """
        if not isinstance(image, np.ndarray):
            raise ValueError(f'Input image should be with type `numpy.ndarray`!')
        if image.dtype != np.uint8:
            raise ValueError(f'Input image should be with dtype `numpy.uint8`!')

        if image.ndim != 3 or image.shape[2] not in [1, 3]:
            raise ValueError(f'Input should be with shape [height, width, channel], '
                             f'where channel equals to 1 or 3!\n'
                             f'But {image.shape} is received!')
        if image.shape[2] == 1 and self.G.image_channels == 3:
            image = np.tile(image, (1, 1, 3))
        if image.shape[2] != self.G.image_channels:
            raise ValueError(f'Number of channels of input image, which is '
                             f'{image.shape[2]}, is not supported by the current '
                             f'inverter, which requires {self.G.image_channels} '
                             f'channels!')

        if self.G.image_channels == 3 and self.G.channel_order == 'BGR':
            image = image[:, :, ::-1]
        if image.shape[1:3] != [self.G.resolution, self.G.resolution]:
            image = cv2.resize(image, (self.G.resolution, self.G.resolution))
        image = image.astype(np.float32)
        image = image / 255.0 * (self.G.max_val - self.G.min_val) + self.G.min_val
        image = image.astype(np.float32).transpose(2, 0, 1)
        return image

    def invert(self, *args, **kwargs):
        raise NotImplementedError(f'Should be implemented in derived class!')
