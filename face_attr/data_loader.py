from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import torch
import os
import random


class CelebA_HQ(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, attr_path, selected_attrs, transform):
        """Initialize and preprocess the CelebA dataset."""
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.attr2idx = {}
        self.idx2attr = {}

        self.current_dataset = []
        self.preprocess()

    def preprocess(self):
        """Preprocess the CelebA attribute file."""
        raise NotImplementedError(f"Should be implemented in derived class!")

    def __getitem__(self, index):
        """
        Return one image, its corresponding attribute label and its filename.
        Such as (image, label, filename).
        image.size (1, 3, 1024, 1024)
        label.size (1, 5), where 5 is the number of selected attributes.
        filename is a string: ''
        """
        dataset = self.current_dataset
        filename, label = dataset[index]
        image = Image.open(os.path.join(self.image_dir, filename))
        # image = cv2.imread(os.path.join(self.image_dir, filename))
        # return image[:, :, ::-1], torch.FloatTensor(label)
        return self.transform(image), torch.FloatTensor(label), filename


class CelebA_HQ_train(CelebA_HQ):

    def __init__(self, image_dir, attr_path, selected_attrs, transform, **kwargs):
        super().__init__(image_dir, attr_path, selected_attrs, transform)

    def preprocess(self):
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        # random.seed(1234)
        # random.shuffle(lines)
        for i, line in enumerate(lines):
            if (i + 1) > 2000:
                split = line.split()
                filename = split[0]
                values = split[1:]

                label = []
                for attr_name in self.selected_attrs:
                    idx = self.attr2idx[attr_name]
                    label.append(values[idx] == '1')
                self.current_dataset.append([filename, label])

        print('Finished preprocessing the CelebA_HQ_train dataset...')

    def __len__(self):
        """Return the number of images."""
        return len(self.current_dataset)


class CelebA_HQ_test(CelebA_HQ):

    def __init__(self, image_dir, attr_path, selected_attrs, transform, nums=None):
        self.nums = nums
        super().__init__(image_dir, attr_path, selected_attrs, transform)

    def preprocess(self):
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(42)
        random.shuffle(lines)
        nums = 2000 if self.nums is None else self.nums
        for i, line in enumerate(lines):
            if (i + 1) <= nums:
                split = line.split()
                filename = split[0]
                values = split[1:]

                label = []
                for attr_name in self.selected_attrs:
                    idx = self.attr2idx[attr_name]
                    label.append(values[idx] == '1')
                self.current_dataset.append([filename, label])
            else:
                break

        print('Finished preprocessing the CelebA_HQ_test dataset...')

    def __len__(self):
        """Return the number of images."""
        return len(self.current_dataset)


def get_loader(image_dir,
               attr_path,
               selected_attrs,
               num_workers=1,
               batch_size=1,
               state='test',
               nums=2000,
               transform=None):
    """Build and return a data loader."""
    if transform is None:
        transform = []
        transform.append(T.ToTensor())
        # transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)

    if state == 'train':
        dataset = CelebA_HQ_train(image_dir, attr_path, selected_attrs, transform)
    else:
        dataset = CelebA_HQ_test(image_dir, attr_path, selected_attrs, transform, nums)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers)
    return data_loader
