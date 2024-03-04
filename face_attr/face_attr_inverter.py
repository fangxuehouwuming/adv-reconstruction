import sys
import numpy as np
from tqdm import tqdm
import torch

from base_inverter import BaseStyleGANInverter

from StarGAN.stargan import Generator


def _get_tensor_value(tensor):
    """Gets the value of a torch Tensor."""
    return tensor.cpu().detach().numpy()


def load_stargan():
    starG = Generator(conv_dim=64, c_dim=5, repeat_num=6)
    G_path = "./StarGAN/stargan_celeba_256/200000-G.ckpt"
    # G_path = "./StarGAN/200000-G.ckpt"

    starG.load_state_dict(torch.load(G_path, map_location=lambda storage, loc: storage))
    # resize = torch.nn.Upsample(size=(128, 128), mode='bilinear').cuda()
    resize = torch.nn.Upsample(size=(256, 256), mode='bilinear').cuda()

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


class FaceAttrInverter(BaseStyleGANInverter):

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
        super(FaceAttrInverter, self).__init__(model_name, logger)
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.regularization_loss_weight = regularization_loss_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.epsilon = epsilon
        assert self.reconstruction_loss_weight > 0

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

    def invert(self, img_idx, image, label, num_viz=0):
        """Inverts the given image to a latent code.

        Basically, this function is based on gradient descent algorithm.

        Returns:
            viz_results: 原始图像x; G(z_0); G(z_n)

            starG_results: 对于每个编辑属性, 包含Fake(x)和Fake(G(z_n)); 例如, 5个属性, 则包含10个图像
        """
        loss_a = 0.0
        loss_b = 0.0
        # =================================================================================== #
        #                                                                                     #
        #                   1. Stage(a): inverting real faces into latent space               #
        #                                                                                     #
        # =================================================================================== #

        # print(f'Stage(a) begins!')

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
        pbara = tqdm(range(1, self.iteration + 1), leave=True)
        # pbar = tqdm(range(1, self.iteration + 1), leave=False)

        for step in pbara:
            loss = 0.0

            # Reconstruction loss.
            if self.reconstruction_loss_weight:
                x_rec = self.G.net.synthesis(z)  # x_rec = x_init_inv = G(z_0)
                loss_pix = torch.mean((x - x_rec)**2)
                loss = loss + loss_pix * self.reconstruction_loss_weight
                log_message = f'Stage(a): loss_pix: {_get_tensor_value(loss_pix):.3f}'

            # Perceptual loss.
            if self.perceptual_loss_weight:
                x_feat = self.F.net(x)
                x_rec_feat = self.F.net(x_rec)
                loss_feat = torch.mean((x_feat - x_rec_feat)**2)
                loss = loss + loss_feat * self.perceptual_loss_weight
                log_message += f', loss_feat: {_get_tensor_value(loss_feat):.3f}'

            log_message += f', loss: {_get_tensor_value(loss):.3f}'
            pbara.set_description_str(log_message)
            if self.logger:
                self.logger.debug(f'ImgID: {img_idx:05d}, '
                                  f'Stage(a), '
                                  f'Step: {step:05d}, '
                                  #   f'lr: {self.learning_rate:.2e}, '
                                  f'{log_message}')

            if step == self.iteration:
                loss_a = loss

            # Do optimization.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ====================================================================================================== #
        #                                                                                                        #
        #     2. Stage(b):  adversarially searching in the latent space for fooling the target DeepFake model    #
        #                                                                                                        #
        # ====================================================================================================== #

        # print(f'\nStage(b) begins!')

        #load stargan
        starG, resize = load_stargan()
        starG.to(self.run_device)
        # starG.eval()
        stargan_results = []

        temp_z = z.detach().clone()  # temp_z is the output from stage(a)
        pbarb = tqdm(range(1, self.iteration + 1), leave=True)
        # pbar = tqdm(range(1, self.iteration + 1), leave=False)

        epsilon = self.epsilon

        # TODO: which opt ???
        # optimizer = torch.optim.Adam([z], lr=self.learning_rate)
        # optimizer = CustomOptimizer([z], self.learning_rate, epsilon, temp_z)
        # optimizer = ClippedAdam([z], self.learning_rate, epsilon)

        for step in pbarb:
            loss_b = 0.0

            # add = torch.clamp(z - temp_z, -epsilon, epsilon)
            # z_add = temp_z + add

            x_rec = self.G.net.synthesis(z)
            # x_rec = self.G.net.synthesis(z_add)

            # Reconstruction loss.
            if self.reconstruction_loss_weight:
                loss_pix = torch.mean((x - x_rec)**2)
                loss_b = loss_b + loss_pix * self.reconstruction_loss_weight
                log_message = f'Stage(b): loss_pix: {_get_tensor_value(loss_pix):.3f}'

            # Perceptual loss.
            if self.perceptual_loss_weight:
                x_feat = self.F.net(x)
                x_rec_feat = self.F.net(x_rec)
                loss_feat = torch.mean((x_feat - x_rec_feat)**2)
                loss_b = loss_b + loss_feat * self.perceptual_loss_weight
                log_message += f', loss_feat: {_get_tensor_value(loss_feat):.3f}'

            # Adversarial loss
            if self.adversarial_loss_weight:
                out_recs = []
                out_oris = []
                loss_adv = 0
                for c_trg in label:
                    out_rec = starG(resize(x_rec), c_trg)
                    out_ori = starG(resize(x), c_trg)
                    # out_rec = starG(x_rec, c_trg)
                    # out_ori = starG(x, c_trg)

                    if step == 1:
                        # print(out_rec.shape, out_ori.shape, x_rec.shape, x.shape)
                        pass
                    # out_rec = starG(x_rec, c_trg)
                    # out_ori = starG(x, c_trg)

                    out_recs.append(out_rec)
                    out_oris.append(out_ori)
                    # loss_adv += torch.mean((resize(x) - out_rec)**2)
                    loss_adv += torch.mean((resize(x_rec) - out_rec)**2)

                loss_b = loss_b + loss_adv * self.adversarial_loss_weight
                log_message += f', loss_adv: {_get_tensor_value(loss_adv):.3f}'

            # soft constraint
            # self.soft_constraint_weight = 2.0
            # if self.soft_constraint_weight:
            #     loss_b = loss_b + torch.mean((temp_z - z)**2) * self.soft_constraint_weight

            log_message += f', loss: {_get_tensor_value(loss_b):.3f}'
            pbarb.set_description_str(log_message)
            if self.logger:
                self.logger.debug(f'ImgID: {img_idx:05d}, '
                                  f'Stage(b), '
                                  f'Step: {step:05d}, '
                                  #   f'lr: {self.learning_rate:.2e}, '
                                  f'{log_message}')

            if step == self.iteration:
                for num in range(len(out_recs)):
                    stargan_results.append(
                        0.5 * 255. *
                        (out_recs[num][0] + 1).detach().cpu().numpy().transpose(1, 2, 0))
                for num in range(len(out_oris)):
                    stargan_results.append(
                        0.5 * 255. *
                        (out_oris[num][0] + 1).detach().cpu().numpy().transpose(1, 2, 0))

            # Do optimization.
            z_old = z.clone()
            optimizer.zero_grad()
            loss_b.backward()
            optimizer.step()

            # Clip latent code z to proper range.
            # with torch.no_grad():

            # opt2
            # add = torch.clamp(z - temp_z - self.learning_rate * z.grad, -epsilon, epsilon)
            # z.data = temp_z.data + add

            # opt1
            # add = torch.clamp(z - z_old, -epsilon, epsilon)
            # z.data = z_old.data + add

            # opt4
            add = torch.clamp(z - temp_z, -epsilon, epsilon)
            z.data = temp_z.data + add

        x_inv = self.G.net.synthesis(z)
        viz_results.append(self.G.postprocess(_get_tensor_value(x_inv))[0])
        loss_result = {
            "ImgID": img_idx,
            "loss_a": loss_a.item(),
            "loss_b": loss_b.item(),
            "loss_a+b": (loss_a + loss_b).item()
        }
        return _get_tensor_value(z), viz_results, stargan_results, loss_result

    def easy_invert(self, img_idx, image, label, num_viz=0):
        """Wraps functions `preprocess()` and `invert()` together."""
        print("******************** {} is working ********************".format(
            self.__class__.__name__))
        return self.invert(img_idx, self.preprocess(image), label, num_viz)
        # return self.invert(img_idx, image, label, num_viz)
