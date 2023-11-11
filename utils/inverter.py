# python 3.7
"""Utility functions to invert a given image back to a latent code."""
import sys
from tqdm import tqdm
import cv2
import numpy as np
from PIL import Image
import random
import torchgeometry as tgm

import torch
from torchvision import transforms as T

from models.stylegan_generator import StyleGANGenerator
from models.stylegan_encoder import StyleGANEncoder
from models.perceptual_model import PerceptualModel

sys.path.append('./StarGAN/')
from stargan import Generator

__all__ = ['StyleGANInverter']


def _softplus(x):
    """Implements the softplus function."""
    return torch.nn.functional.softplus(x, beta=1, threshold=10000)


def _get_tensor_value(tensor):
    """Gets the value of a torch Tensor."""
    return tensor.cpu().detach().numpy()


def load_stargan():
    starG = Generator(conv_dim=64, c_dim=5, repeat_num=6)
    G_path = "./StarGAN/stargan_celeba_128/models/200000-G.ckpt"
    starG.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    resize = torch.nn.Upsample(size=(128, 128), mode='bilinear').cuda()
    return starG, resize


class CustomOptimizer(torch.optim.Optimizer):
    '''Custom optimizer for stage(b).'''

    def __init__(self, params, alpha, epsilon, z_fixed):
        if alpha < 0.0:
            raise ValueError("Invalid learning rate: {}".format(alpha))
        if epsilon < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(epsilon))
        defaults = dict(alpha=alpha, epsilon=epsilon, z_fixed=z_fixed)
        super(CustomOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad_p = p.grad
                z_prime = p.data
                z_fixed = group['z_fixed']
                alpha = group['alpha']
                epsilon = group['epsilon']

                update = z_prime - z_fixed - alpha * grad_p
                update = torch.clamp(update, -epsilon, epsilon)
                p.data = z_fixed + update


class ClippedAdam(torch.optim.Adam):
    '''Clipped Adam optimizer for stage(b).'''

    def __init__(self, params, lr, epsilon, *args, **kwargs):
        super().__init__(params, lr, *args, **kwargs)
        self.epsilon = epsilon

    @torch.no_grad()
    def step(self, closure=None):
        super().step(closure=closure)
        for group in self.param_groups:
            for p in group['params']:
                p.data.clamp_(-self.epsilon, self.epsilon)


class StyleGANInverter(object):
    """Defines the class for StyleGAN inversion.

  Even having the encoder, the output latent code is not good enough to recover
  the target image satisfyingly. To this end, this class optimize the latent
  code based on gradient descent algorithm. In the optimization process,
  following loss functions will be considered:

  (1) Pixel-wise reconstruction loss. (required)
  (2) Perceptual loss. (optional, but recommended)
  (3) Regularization loss from encoder. (optional, but recommended for in-domain
      inversion)

  NOTE: The encoder can be missing for inversion, in which case the latent code
  will be randomly initialized and the regularization loss will be ignored.
  """

    def __init__(self,
                 model_name,
                 learning_rate=1e-2,
                 iteration=100,
                 reconstruction_loss_weight=1.0,
                 perceptual_loss_weight=5e-5,
                 regularization_loss_weight=5.0,
                 adversarial_loss_weight=1.0,
                 epsilon=0.05,
                 logger=None):
        """Initializes the inverter.

    NOTE: Only Adam optimizer is supported in the optimization process.

    Args:
      model_name: Name of the model on which the inverted is based. The model
        should be first registered in `models/model_settings.py`.
      logger: Logger to record the log message.
      learning_rate: Learning rate for optimization. (default: 1e-2)
      iteration: Number of iterations for optimization. (default: 100)
      reconstruction_loss_weight: Weight for reconstruction loss. Should always
        be a positive number. (default: 1.0)
      perceptual_loss_weight: Weight for perceptual loss. 0 disables perceptual
        loss. (default: 5e-5)
      regularization_loss_weight: Weight for regularization loss from encoder.
        This is essential for in-domain inversion. However, this loss will
        automatically ignored if the generative model does not include a valid
        encoder. 0 disables regularization loss. (default: 2.0)
    """
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

        self.learning_rate = learning_rate
        self.iteration = iteration
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.regularization_loss_weight = regularization_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.epsilon = epsilon
        assert self.reconstruction_loss_weight > 0

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

    def get_init_code(self, image):
        """Gets initial latent codes as the start point for optimization.

    The input image is assumed to have already been preprocessed, meaning to
    have shape [self.G.image_channels, self.G.resolution, self.G.resolution],
    channel order `self.G.channel_order`, and pixel range [self.G.min_val,
    self.G.max_val].
    """
        x = image[np.newaxis]
        x = self.G.to_tensor(x.astype(np.float32))
        z = _get_tensor_value(self.E.net(x).view(1, *self.encode_dim))
        return z.astype(np.float32)

    # def invert(self, image, label, num_viz=0):

    #     #load stargan
    #     import sys
    #     sys.path.append('./StarGAN/')
    #     from stargan import Generator
    #     starG = Generator(conv_dim=64, c_dim=5, repeat_num=6)
    #     G_path = "./StarGAN/stargan_celeba_128/models/200000-G.ckpt"
    #     starG.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    #     starG.to(self.run_device)
    #     resize = torch.nn.Upsample(size=(128, 128), mode='bilinear').cuda()

    #     # stage(a) : get the initial latent code and do refinement
    #     print(f'Stage(a) begins!')
    #     x = image[np.newaxis]
    #     x = self.G.to_tensor(x.astype(np.float32))
    #     x.requires_grad = False
    #     init_z = self.get_init_code(image)  # z_0
    #     z = torch.Tensor(init_z).to(self.run_device)
    #     z.requires_grad = True

    #     optimizer = torch.optim.Adam([z], lr=self.learning_rate)

    #     viz_results = []
    #     viz_results.append(self.G.postprocess(_get_tensor_value(x))[0])
    #     x_init_inv = self.G.net.synthesis(z)
    #     viz_results.append(self.G.postprocess(_get_tensor_value(x_init_inv))[0])
    #     pbar = tqdm(range(1, self.iteration + 1), leave=True)

    #     stargan_results = []

    #     for step in pbar:
    #         loss = 0.0

    #         # Reconstruction loss.
    #         x_rec = self.G.net.synthesis(z)  # x_rec = x_init_inv = G(z_0)
    #         loss_pix = torch.mean((x - x_rec)**2)
    #         loss = loss + loss_pix * self.loss_pix_weight
    #         log_message = f'loss_pix: {_get_tensor_value(loss_pix):.3f}'

    #         # Perceptual loss.
    #         if self.loss_feat_weight:
    #             x_feat = self.F.net(x)
    #             x_rec_feat = self.F.net(x_rec)
    #             loss_feat = torch.mean((x_feat - x_rec_feat)**2)
    #             loss = loss + loss_feat * self.loss_feat_weight
    #             log_message += f', loss_feat: {_get_tensor_value(loss_feat):.3f}'

    #         # # Regularization loss.
    #         # if self.loss_reg_weight:
    #         #     z_rec = self.E.net(x_rec).view(1, *self.encode_dim)
    #         #     loss_reg = torch.mean((z - z_rec)**2)
    #         #     loss = loss + loss_reg * self.loss_reg_weight
    #         #     log_message += f', loss_reg: {_get_tensor_value(loss_reg):.3f}'

    #         log_message += f', loss: {_get_tensor_value(loss):.3f}'
    #         pbar.set_description_str(log_message)
    #         if self.logger:
    #             self.logger.debug(f'Stage(a), '
    #                               f'Step: {step:05d}, '
    #                               f'lr: {self.learning_rate:.2e}, '
    #                               f'{log_message}')

    #         # Do optimization.
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #     # stage(b): search a neighbor embedding which still reconstructs faithfully but disables DeepFake
    #     print(f'\nStage(b) begins!')
    #     temp_z = z.detach().clone()  # temp_z is the output from stage(a)
    #     pbar = tqdm(range(1, self.iteration + 1), leave=True)
    #     epsilon = 0.05

    #     # optimizer = CustomOptimizer([z], self.learning_rate, epsilon, temp_z)
    #     # optimizer = ClippedAdam([z], self.learning_rate, epsilon)

    #     for step in pbar:
    #         loss = 0.0

    #         '''NOTE: original code'''
    #         # Reconstruction loss.
    #         add = torch.clamp(z - temp_z, -epsilon, epsilon)
    #         z_add = temp_z + add
    #         x_rec = self.G.net.synthesis(z_add)
    #         loss_pix = torch.mean((x - x_rec)**2)
    #         loss = loss + loss_pix * self.loss_pix_weight
    #         log_message = f'loss_pix: {_get_tensor_value(loss_pix):.3f}'

    #         # Perceptual loss.
    #         if self.loss_feat_weight:
    #             x_feat = self.F.net(x)
    #             x_rec_feat = self.F.net(x_rec)
    #             loss_feat = torch.mean((x_feat - x_rec_feat)**2)
    #             loss = loss + loss_feat * self.loss_feat_weight
    #             log_message += f', loss_feat: {_get_tensor_value(loss_feat):.3f}'

    #         # adversarial loss
    #         out_recs = []
    #         out_oris = []
    #         out_rec = starG(resize(x_rec), label[0])  # (1, 3, 128, 128)
    #         out_ori = starG(resize(x), label[0])  # (1, 3, 128, 128)
    #         out_recs.append(out_rec)
    #         out_oris.append(out_ori)
    #         loss_adv = torch.mean((resize(x) - out_rec)**2)
    #         for c_trg in label[1:]:
    #             out_rec = starG(resize(x_rec), c_trg)
    #             out_ori = starG(resize(x), c_trg)
    #             out_recs.append(out_rec)
    #             out_oris.append(out_ori)
    #             loss_adv += torch.mean((resize(x) - out_rec)**2)
    #         if step == self.iteration:
    #             for num in range(len(out_recs)):
    #                 stargan_results.append(
    #                     0.5 * 255. *
    #                     (out_recs[num][0] + 1).detach().cpu().numpy().transpose(1, 2, 0))
    #             for num in range(len(out_oris)):
    #                 stargan_results.append(
    #                     0.5 * 255. *
    #                     (out_oris[num][0] + 1).detach().cpu().numpy().transpose(1, 2, 0))
    #         loss = loss + loss_adv * 1.0  #self.loss_adv_weight
    #         log_message += f', loss_adv: {_get_tensor_value(loss_adv):.3f}'
    #         # log_message = f'loss_adv: {_get_tensor_value(loss_adv):.3f}'

    #         log_message += f', loss: {_get_tensor_value(loss):.3f}'
    #         pbar.set_description_str(log_message)
    #         if self.logger:
    #             self.logger.debug(f'Stage(b), '
    #                               f'Step: {step:05d}, '
    #                               f'lr: {self.learning_rate:.2e}, '
    #                               f'{log_message}')

    #     x_inv = self.G.net.synthesis(z)
    #     viz_results.append(self.G.postprocess(_get_tensor_value(x_inv))[0])

    #     # viz_results: 原始图像x; G(z_0); G(z_n)
    #     # starG_results: 对于每个编辑属性, 包含Fake(x)和Fake(G(z_n)); 例如, 5个属性, 则包含10个图像
    #     return _get_tensor_value(z), viz_results, stargan_results

    def invert(self, image, label, num_viz=0):
        """Inverts the given image to a latent code.

        Basically, this function is based on gradient descent algorithm.

        Returns:
            viz_results: 原始图像x; G(z_0); G(z_n)

            starG_results: 对于每个编辑属性, 包含Fake(x)和Fake(G(z_n)); 例如, 5个属性, 则包含10个图像
        """

        # =================================================================================== #
        #                                                                                     #
        #                   1. Stage(a): inverting real faces into latent space               #
        #                                                                                     #
        # =================================================================================== #

        print(f'Stage(a) begins!')

        x = image[np.newaxis]
        x = self.G.to_tensor(x.astype(np.float32))
        x.requires_grad = False

        init_z = self.get_init_code(image)  # z_0
        z = torch.Tensor(init_z).to(self.run_device)
        z.requires_grad = True

        # use Adam optimizer to do refinement for z_0
        optimizer = torch.optim.Adam([z], lr=self.learning_rate)

        viz_results = []
        viz_results.append(self.G.postprocess(_get_tensor_value(x))[0])
        x_init_inv = self.G.net.synthesis(z)
        viz_results.append(self.G.postprocess(_get_tensor_value(x_init_inv))[0])
        pbar = tqdm(range(1, self.iteration + 1), leave=True)

        for step in pbar:
            loss = 0.0

            # Reconstruction loss.
            x_rec = self.G.net.synthesis(z)  # x_rec = x_init_inv = G(z_0)
            loss_pix = torch.mean((x - x_rec)**2)
            loss = loss + loss_pix * self.reconstruction_loss_weight
            log_message = f'loss_pix: {_get_tensor_value(loss_pix):.3f}'

            # Perceptual loss.
            if self.perceptual_loss_weight:
                x_feat = self.F.net(x)
                x_rec_feat = self.F.net(x_rec)
                loss_feat = torch.mean((x_feat - x_rec_feat)**2)
                loss = loss + loss_feat * self.perceptual_loss_weight
                log_message += f', loss_feat: {_get_tensor_value(loss_feat):.3f}'

            log_message += f', loss: {_get_tensor_value(loss):.3f}'
            pbar.set_description_str(log_message)
            if self.logger:
                self.logger.debug(f'Stage(a), '
                                  f'Step: {step:05d}, '
                                  f'lr: {self.learning_rate:.2e}, '
                                  f'{log_message}')

            # Do optimization.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ====================================================================================================== #
        #                                                                                                        #
        #     2. Stage(b):  adversarially searching in the latent space for fooling the target DeepFake model    #
        #                                                                                                        #
        # ====================================================================================================== #

        print(f'\nStage(b) begins!')

        #load stargan
        starG, resize = load_stargan()
        starG.to(self.run_device)
        stargan_results = []

        temp_z = z.detach().clone()  # temp_z is the output from stage(a)
        pbar = tqdm(range(1, self.iteration + 1), leave=True)
        epsilon = self.epsilon

        # TODO: which opt ???
        # optimizer = torch.optim.Adam([z], lr=self.learning_rate)
        # optimizer = CustomOptimizer([z], self.learning_rate, epsilon, temp_z)
        # optimizer = ClippedAdam([z], self.learning_rate, epsilon)

        for step in pbar:
            loss = 0.0

            # Reconstruction loss.
            add = torch.clamp(z - temp_z, -epsilon, epsilon)
            z_add = temp_z + add
            x_rec = self.G.net.synthesis(z_add)
            loss_pix = torch.mean((x - x_rec)**2)
            loss = loss + loss_pix * self.reconstruction_loss_weight
            log_message = f'loss_pix: {_get_tensor_value(loss_pix):.3f}'

            # Perceptual loss.
            if self.perceptual_loss_weight:
                x_feat = self.F.net(x)
                x_rec_feat = self.F.net(x_rec)
                loss_feat = torch.mean((x_feat - x_rec_feat)**2)
                loss = loss + loss_feat * self.perceptual_loss_weight
                log_message += f', loss_feat: {_get_tensor_value(loss_feat):.3f}'

            # Adversarial loss
            if self.adversarial_loss_weight:
                out_recs = []
                out_oris = []
                loss_adv = 0
                for c_trg in label:
                    out_rec = starG(resize(x_rec), c_trg)
                    out_ori = starG(resize(x), c_trg)
                    out_recs.append(out_rec)
                    out_oris.append(out_ori)
                    loss_adv += torch.mean((resize(x) - out_rec)**2)
                loss = loss + loss_adv * self.adversarial_loss_weight
                log_message += f', loss_adv: {_get_tensor_value(loss_adv):.3f}'

                # save the last stargan results
                if step == self.iteration:
                    for num in range(len(out_recs)):
                        stargan_results.append(
                            0.5 * 255. *
                            (out_recs[num][0] + 1).detach().cpu().numpy().transpose(1, 2, 0))
                    for num in range(len(out_oris)):
                        stargan_results.append(
                            0.5 * 255. *
                            (out_oris[num][0] + 1).detach().cpu().numpy().transpose(1, 2, 0))

            log_message += f', loss: {_get_tensor_value(loss):.3f}'
            pbar.set_description_str(log_message)
            if self.logger:
                self.logger.debug(f'Stage(b), '
                                  f'Step: {step:05d}, '
                                  f'lr: {self.learning_rate:.2e}, '
                                  f'{log_message}')

            # Do optimization.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        x_inv = self.G.net.synthesis(z)
        viz_results.append(self.G.postprocess(_get_tensor_value(x_inv))[0])
        return _get_tensor_value(z), viz_results, stargan_results

    def easy_invert(self, image, label, num_viz=0):
        """Wraps functions `preprocess()` and `invert()` together."""
        return self.invert(self.preprocess(image), label, num_viz)
