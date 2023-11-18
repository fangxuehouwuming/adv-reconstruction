'''
For face attribute editing, we consider the L2-norm distance and compute the mean confidence difference (MCD) with an attribute recognition model.
'''
import os

import torch
import torchvision.transforms as transforms
from resnet import resnet50, resnet101
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 224
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])


def L2_distance(original, reconstructed):
    return torch.sqrt(torch.sum((original - reconstructed)**2))


def MCD(model, original, reconstructed):
    original_confidence = model(original.unsqueeze(0))  # 0.2565
    reconstructed_confidence = model(reconstructed.unsqueeze(0))  # 0.2473
    confidence_difference = torch.abs(original_confidence - reconstructed_confidence)
    # confidence_difference = original_confidence - reconstructed_confidence
    return confidence_difference.squeeze(0)


def get_recognition_model(model_name="resnet50", model_path="metrics/pretrain/resnet50_65k.pth"):
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
    folder = "./results/face_attr_inversion"
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

    print("L2-norm distance:")
    sum_distance = 0
    for img_id in x_imgs:
        distance = L2_distance(x_imgs[img_id], gzn_imgs[img_id])
        sum_distance += distance
        print(img_id, distance.item())
    print("mean:", sum_distance.item() / len(x_imgs))

    print("MCD:")
    model_name = "resnet50"
    model_path = "metrics/pretrain/resnet50_130k.pth"
    regconition_model = get_recognition_model(model_name=model_name, model_path=model_path)
    regconition_model.eval()
    sum_mcd = 0
    for img_id in gzn_imgs:
        per_mcd = 0
        for attr in fake_gzn_imgs[img_id]:
            original = gzn_imgs[img_id]
            reconstructed = fake_gzn_imgs[img_id][attr]
            mcd = MCD(regconition_model, original, reconstructed)[attr2idx[attr]]
            per_mcd += mcd
            print(f'Image ID: {img_id}, Attribute: {attr}, MCD: {mcd}')
        # sum_mcd += per_mcd / len(fake_gzn_imgs[img_id])
        sum_mcd += per_mcd
    print("mean:", sum_mcd / len(gzn_imgs))
