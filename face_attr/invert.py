# python 3.6
import os
import csv
import argparse
import numpy as np

from torchvision import transforms
from tqdm import tqdm

from utils.logger import setup_logger
from utils.visualizer import save_image

from data_loader import get_loader

from face_attr_inverter import FaceAttrInverter


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        # default="styleganinv_ffhq256",
        default="styleganinv_celebahq256",
        help="Name of the GAN model.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=
        "./results/face_attr_hq256encoder_hq256imgsize_256stargan_100iter_adv2_epsilon0.05",
    )

    # parser.add_argument(
    #     "--output_ori_dir",
    #     type=str,
    #     default=
    #     "./results/face_attr_hq256encoder_hq256imgsize_256stargan_100iter_adv2_epsilon0.05/original",
    #     help="Directory to save the results. If not specified, "
    #     "`./results/face_attr_inversion/original` "
    #     "will be used by default.",
    # )
    # parser.add_argument(
    #     "--output_fake_dir",
    #     type=str,
    #     default=
    #     "./results/face_attr_hq256encoder_hq256imgsize_256stargan_100iter_adv2_epsilon0.05/stargan",
    #     help="Directory to save the results. If not specified, "
    #     "`./results/face_attr_inversion/stargan` "
    #     "will be used by default.",
    # )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for optimization. (default: 0.01)",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=100,
        help="Number of optimization iterations. (default: 100)",
    )
    parser.add_argument(
        "--num_results",
        type=int,
        default=5,
        help="Number of intermediate optimization results to "
        "save for each sample. (default: 5)",
    )

    parser.add_argument(
        "--reconstruction_loss_weight",
        type=float,
        default=1.0,
        help="The reconstruction loss scale for optimization. "
        "(default: 1.0)",
    )
    parser.add_argument(
        "--perceptual_loss_weight",
        type=float,
        default=5e-5,
        help="The perceptual loss scale for optimization. "
        "(default: 5e-5)",
    )
    parser.add_argument(
        "--regularization_loss_weight",
        type=float,
        default=2,
        help="The regularization loss scale for optimization. "
        "(default: 5.0)",
    )
    parser.add_argument(
        "--adversarial_loss_weight",
        type=float,
        default=2.0,
        help="The adversarial loss scale for optimization. "
        "(default: 2.0)",
    )

    parser.add_argument(
        "--viz_size",
        type=int,
        default=256,
        help="Image size for visualization. (default: 256)",
    )
    parser.add_argument("--gpu_id",
                        type=str,
                        default="0",
                        help="Which GPU(s) to use. (default: `0`)")
    parser.add_argument(
        "--celeba_image_dir",
        type=str,
        # default="input celeba-hq dataset path here",
        default="F:\dataset\CelebAMask-HQ\CelebA-HQ-img",
        help="path for celeb data",
    )
    parser.add_argument(
        "--attr_path",
        type=str,
        # default="input celeba-hq attribute annatations txt here",
        default="F:\dataset\CelebAMask-HQ\CelebAMask-HQ-attribute-anno.txt",
        help="path for celeb anno",
    )
    parser.add_argument(
        "--selected_attrs",
        "--list",
        nargs="+",
        help="selected attributes for the CelebA dataset",
        default=["Black_Hair", "Blond_Hair", "Brown_Hair", "Male", "Young"],
    )
    parser.add_argument("--c_dim",
                        type=int,
                        default=5,
                        help="dimension of domain labels (1st dataset)")
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    output_dir = args.output_dir
    output_ori_dir = output_dir + "/original"
    output_fake_dir = output_dir + "/stargan"

    # 如果不存在文件夹则创建
    os.makedirs(output_ori_dir, exist_ok=True)
    os.makedirs(output_fake_dir, exist_ok=True)

    # 在ouput_ori_dir的上一级目录下创建日志文件
    logger = setup_logger(output_dir, "inversion.log", "inversion_logger")

    csv_path = "./results/face_attr_inversion/loss_results.csv"
    fieldnames = ["ImgID", "loss_a", "loss_b", "loss_a+b"]
    # with open(csv_path, 'a', newline='') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     csvfile.close()

    # ================================================ #
    #
    #                   1. load model                  #
    #
    # ================================================ #
    logger.info(f"Loading model.")
    inverter = FaceAttrInverter(
        args.model_name,
        learning_rate=args.learning_rate,
        iteration=args.num_iterations,
        reconstruction_loss_weight=args.reconstruction_loss_weight,
        perceptual_loss_weight=args.perceptual_loss_weight,
        regularization_loss_weight=args.regularization_loss_weight,
        adversarial_loss_weight=args.adversarial_loss_weight,
        epsilon=0.05,
        logger=logger,
    )

    # ================================================== #
    #
    #                   2. load dataset                  #
    #
    # ================================================== #
    # load celeba
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 224
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    data_loader = get_loader(args.celeba_image_dir,
                             args.attr_path,
                             args.selected_attrs,
                             num_workers=4,
                             batch_size=1,
                             state='test',
                             nums=500,
                             transform=transform)
    # print('len(data_loader):', len(data_loader))

    # ================================================== #
    #
    #                   2. start invert                  #
    #
    # ================================================== #

    for img_idx, (x_real, c_org,
                  filename) in enumerate(tqdm(data_loader, desc="Outer Loop", leave=True)):
        '''
        注意dataloader的输出:
        '''

        image = (255 * x_real[0].numpy().transpose(1, 2, 0)).astype(np.uint8)

        hair_color_indices = []  # Indices of selected hair colors.
        for i, attr_name in enumerate(args.selected_attrs):
            if attr_name in ["Black_Hair", "Blond_Hair", "Brown_Hair", "Gray_Hair"]:
                hair_color_indices.append(i)
        label = []
        for i in range(args.c_dim):
            c_trg = c_org.clone()
            if i in hair_color_indices:  # Set one hair color to 1 and the rest to 0.
                c_trg[:, i] = 1
                for j in hair_color_indices:
                    if j != i:
                        c_trg[:, j] = 0
            else:
                c_trg[:, i] = c_trg[:, i] == 0  # Reverse attribute value.
            label.append(c_trg.cuda())

            # c_org:
            # tensor([[0., 0., 0., 0., 1.]])
            # label:
            # tensor([[1., 0., 0., 0., 1.]], device='cuda:0')
            # tensor([[0., 1., 0., 0., 1.]], device='cuda:0')
            # tensor([[0., 0., 1., 0., 1.]], device='cuda:0')
            # tensor([[0., 0., 0., 1., 1.]], device='cuda:0')
            # tensor([[0., 0., 0., 0., 0.]], device='cuda:0')
        x_real = x_real[0].numpy()
        code, viz_results, stargan_results, loss_result = inverter.easy_invert(
            img_idx, image, label, num_viz=args.num_results)

        # ================================================ #
        #
        #                   3.save result                  #
        #
        # ================================================ #

        image_name = os.path.splitext(os.path.basename(filename[0]))[0]

        # viz_results: 原始图像x; G(z_0); G(z_n)
        save_image(f"{output_ori_dir}/{image_name}_x.png", viz_results[0])
        save_image(f"{output_ori_dir}/{image_name}_G(z0).png", viz_results[1])
        save_image(f"{output_ori_dir}/{image_name}_G(zn).png", viz_results[-1])
        os.makedirs(f"{output_ori_dir}", exist_ok=True)

        # starG_results: 对于每个编辑属性, 包含Fake(x)和Fake(G(z_n)); 例如, 5个属性, 则包含10个图像
        for num in range(len(args.selected_attrs)):
            save_image(f"{output_fake_dir}/{image_name}_Fake(G(zn))_{args.selected_attrs[num]}.png",
                       stargan_results[num])
            save_image(
                f"{output_fake_dir}/{image_name}_Fake(x)_{args.selected_attrs[num]}.png",
                stargan_results[num + len(args.selected_attrs)],
            )

        # with open(csv_path, 'a', newline='') as csvfile:
        #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        #     writer.writerow(loss_result)
        #     csvfile.close()

        print('\n')
        # break


if __name__ == "__main__":
    main()
