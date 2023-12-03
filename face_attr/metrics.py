'''
For face attribute editing, we consider the L2-norm distance and compute the mean confidence difference (MCD) with an attribute recognition model.
'''
import os
import sys

import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

from ResNet.resnet import resnet50, resnet101

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 224
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def L2_distance(original, reconstructed):
    '''pixel L2-norm distance
    '''
    # original = (255 * original.numpy().transpose(1, 2, 0)).astype(np.uint8)
    # reconstructed = (255 * reconstructed.numpy().transpose(1, 2, 0)).astype(np.uint8)
    # return np.sqrt(np.sum((original - reconstructed)**2))

    # return torch.sqrt(torch.functional.F.mse_loss(original, reconstructed))
    return torch.functional.F.mse_loss(original, reconstructed)
    # return torch.sqrt(torch.sum((original - reconstructed)**2))


def confidence_difference(original_confidence, reconstructed_confidence):
    confidence_difference = torch.abs(original_confidence - reconstructed_confidence)
    # confidence_difference = original_confidence - reconstructed_confidence
    return confidence_difference


def get_recognition_model(model_name="resnet50", model_path=None):
    if model_name == "resnet50":
        model = resnet50(num_classes=5)
    if model_name == "resnet101":
        model = resnet101(num_classes=5)
    model.load_state_dict(torch.load(model_path))
    return model


def get_x_gz0_gzn_imgs(folder):
    x_imgs = {}
    gz0_imgs = {}
    gzn_imgs = {}
    for filename in os.listdir(folder):
        # filename: 0001_G(zn).png, 0001_x.png
        parts = filename.split('_')
        try:
            img_id = parts[0]
            img_type = parts[1].split('.')[0]
        except:
            continue

        img = transform(Image.open(os.path.join(folder, filename)))

        if img_type == "x" and img_id not in x_imgs:
            x_imgs[img_id] = img
        if img_type == "G(z0)" and img_id not in gz0_imgs:
            gz0_imgs[img_id] = img
        if img_type == "G(zn)" and img_id not in gzn_imgs:
            gzn_imgs[img_id] = img

    return x_imgs, gz0_imgs, gzn_imgs


def get_fake_x_gzn_imgs(fake_folder):
    fake_x_imgs = {}
    fake_gzn_imgs = {}
    for filename in os.listdir(fake_folder):
        parts = filename.split('_')
        img_id = parts[0]
        img_type = parts[1]
        try:
            img_attr = parts[2] + '_' + parts[3].split('.')[0]
        except:
            img_attr = parts[2].split('.')[0]

        img = transform(Image.open(os.path.join(fake_folder, filename)))

        if img_type == "Fake(x)":
            if img_id not in fake_x_imgs:
                fake_x_imgs[img_id] = {}
            fake_x_imgs[img_id][img_attr] = img
        if img_type == "Fake(G(zn))":
            if img_id not in fake_gzn_imgs:
                fake_gzn_imgs[img_id] = {}
            fake_gzn_imgs[img_id][img_attr] = img

    return fake_x_imgs, fake_gzn_imgs


# label: ["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"]
if __name__ == "__main__":
    folder = "./results/face_attr_inversion/original/"
    fake_folder = "./results/face_attr_inversion/stargan/"
    attr2idx = {"Black_Hair": 0, "Blond_Hair": 1, "Brown_Hair": 2, "Male": 3, "Young": 4}
    idx2attr = {0: "Black_Hair", 1: "Blond_Hair", 2: "Brown_Hair", 3: "Male", 4: "Young"}

    # x_imgs:
    # {'2553': tensor([[[0.6314, 0....0.7490]]])}
    # gzn_imgs:
    # {'2553': tensor([[[0.6314, 0....0.7490]]])}
    x_imgs, gz0_imgs, gzn_imgs = get_x_gz0_gzn_imgs(folder)
    # fake_gzn_imgs:
    # {
    #     '2553': {
    #         'Black_Hair': tensor([[[0.6314, 0....0.7490]]])
    #         'Blond_Hair': ...,
    #         'Brown_Hair': ...,
    #         'Male': ...,
    #         'Young': ...,
    #     }
    # }
    fake_x_imgs, fake_gzn_imgs = get_fake_x_gzn_imgs(fake_folder)
    '''
    The L2-norm distance between the reconstructed data and StarGAN's output
    '''
    print("L2-norm distance:")
    sum_distance = 0
    for img_id in gzn_imgs:
        per_distance = 0
        for attr in fake_gzn_imgs[img_id]:
            distance = L2_distance(gzn_imgs[img_id], fake_gzn_imgs[img_id][attr])
            per_distance += distance
        sum_distance += per_distance / len(fake_gzn_imgs[img_id])
        # sum_distance += per_distance
        # print(img_id, distance.item())
    print("mean:", sum_distance.item() / len(gzn_imgs))

    del x_imgs, gz0_imgs, fake_x_imgs
    '''
    The MCD between the reconstructed data and StarGAN's output
    '''
    print("MCD:")
    model_name = "resnet50"
    model_path = "./ResNet/pretrain/resnet50_65k.pth"
    regconition_model = get_recognition_model(model_name=model_name,
                                              model_path=model_path).to("cuda")
    # regconition_model.cuda()
    regconition_model.eval()
    sum_mcd = 0
    cnt = 0
    # torch.cuda.empty_cache()
    with torch.no_grad():
        for img_id in gzn_imgs:
            # if cnt == 10:
            #     break
            cnt += 1
            per_mcd = 0
            for attr in fake_gzn_imgs[img_id]:
                original = gzn_imgs[img_id].to("cuda")
                reconstructed = fake_gzn_imgs[img_id][attr].to("cuda")

                ori_confi = regconition_model(original.unsqueeze(0))  # 0.2565
                rec_confi = regconition_model(reconstructed.unsqueeze(0))  # 0.2473
                confi_dif = confidence_difference(ori_confi, rec_confi)  # 0.0092
                mcd = confi_dif.squeeze(0)[attr2idx[attr]]

                per_mcd += mcd
                # print(f'Image ID: {img_id}, Attribute: {attr}, MCD: {mcd}')
                del original, reconstructed, mcd, ori_confi, rec_confi, confi_dif
                torch.cuda.empty_cache()
            sum_mcd += per_mcd / len(fake_gzn_imgs[img_id])
            # sum_mcd += per_mcd
    print("mean:", sum_mcd.item() / len(gzn_imgs))
    # print("mean:", sum_mcd.item() / cnt)
'''
L2-norm distance: 20
MCD: mean: 0.04093208758283682
'''

# TODO: L2-norm distance ????
