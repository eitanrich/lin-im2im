import os
import torch
from torchvision.datasets import CelebA, CIFAR10, LSUN, ImageFolder
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from utils import CropTransform
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image

# Change the below to the actual dataset root folders
celeba_root = 'datasets/CelebA'
ffhq_root = 'datasets/FFHQ'
shoes_root = 'datasets/edges2shoes'


class Shoes(Dataset):
    """
    Dataset format is the same as used in pix2pix. We take only trainB and testB.
    """
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.im_list = [f for f in os.listdir(os.path.join(root_dir, split+'B')) if f.endswith('jpg')]
        print('Got {} shoes in split {}.'.format(len(self.im_list), split))

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = os.path.join(self.root_dir, self.split+'B', self.im_list[idx])
        image = Image.open(img_path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


class FFHQ(Dataset):
    """
    FFHQ folder should contain images1024x1024 and thumbnails128x128
    """
    def __init__(self, root_dir, split='train', transform=None, use_thumbnails=False):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        self.use_thumbnails = use_thumbnails
        self.split_ranges = {'train': (0, 60000), 'test': (60000, 70000)}

    def __len__(self):
        return self.split_ranges[self.split][1] - self.split_ranges[self.split][0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        subfolder = 'thumbnails128x128' if self.use_thumbnails else 'images1024x1024'
        img_name = os.path.join(self.root_dir, subfolder, '%05i.png' % (idx+self.split_ranges[self.split][0]))
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)
        return image


def load_data(dataset, num_samples=None, w=128, shuffle=True, has_cls=False):
    if num_samples:
        if shuffle:
            dataset = random_split(dataset, [num_samples, len(dataset)-num_samples])[0]
        else:
            dataset = Subset(dataset, np.arange(num_samples))
    loader = DataLoader(dataset, shuffle=shuffle, num_workers=8)

    if has_cls:
        return np.vstack([x.numpy() for x, _ in tqdm(loader)]).transpose([0, 2, 3, 1]).reshape(-1, w*w*3)

    return np.vstack([x.numpy() for x in tqdm(loader)]).transpose([0, 2, 3, 1]).reshape(-1, w*w*3)


def get_ffhq_data(split='train', num_samples=None, w=128, shuffle=True):
    ffhq = FFHQ(ffhq_root, split=split, transform=transforms.Compose([transforms.Resize(w), transforms.ToTensor()]),
                use_thumbnails=(w <= 128))
    return load_data(ffhq, num_samples, w, shuffle)


def get_celeba_data(split='train', num_samples=None, w=128, attr_num=None, attr_value=None, shuffle=True):
    celeba = CelebA(root=celeba_root, split=split,  download=False, target_type='attr',
                    transform=transforms.Compose([CropTransform((25, 50, 25+128, 50+128)),
                                                  transforms.Resize(w),
                                                  transforms.ToTensor()]))
    return load_data(celeba, num_samples, w, shuffle, has_cls=True)


def get_shoes_data(split='train', num_samples=None, w=128, shuffle=True):
    shoes = Shoes(shoes_root, split=split, transform=transforms.Compose([transforms.CenterCrop((256, 256)),
                                                                         transforms.Resize((w, w)),
                                                                         transforms.ToTensor()]))
    return load_data(shoes, num_samples, w, shuffle)


def true_transform(X, ttype='identity', w=128):
    """
    Apply a synthetic transformation to a set of images
    :param X: Images (ch last) flattened - each image as row vector in X
    :param ttype: The required transformation
    :param w: The image resolution (w=h)
    :return: Transformed images
    """
    X = X.reshape(-1, w, w, 3)

    if ttype == 'rot90':
        X = np.rot90(X, k=1, axes=(1, 2))

    elif ttype == 'inpaint':
        mask = cv2.imread('data/inpaint_mask_simple.png').astype(np.float32)/255.0
        # mask = cv2.imread('data/inpaint_mask.png').astype(np.float32)/255.0
        # mask[:, 64:, :] = 1.0 - mask[:, 64:, :]
        if not mask.shape[0] == w:
            mask = cv2.resize(mask, (w, w), interpolation=cv2.INTER_NEAREST)
        X = X.copy() * mask.reshape(1, w, w, 3)

    elif ttype == 'vflip':
        X = X[:, ::-1]

    elif ttype == 'colorize':
        X = np.repeat(np.mean(X, axis=3, keepdims=True), 3, axis=3)

    elif ttype == 'edges':
        ksize = 1 if w == 64 else 3
        X = np.stack([cv2.Laplacian(X[i], cv2.CV_32F, ksize=ksize) for i in range(X.shape[0])])

    elif ttype == 'Canny-edges':
        edges = np.stack([cv2.Canny((np.mean(X[i], axis=2)*255.0).astype(np.uint8), 80, 200) for i in range(X.shape[0])])
        X = np.repeat(np.expand_dims(edges.astype(np.float32)*(1.0/255.0), 3), 3, axis=3)

    elif ttype == 'super-res':
        X = np.stack([cv2.resize(cv2.resize(X[i], (w//8, w//8), interpolation=cv2.INTER_LINEAR), (w, w),
                                 interpolation=cv2.INTER_LINEAR) for i in range(X.shape[0])])
    elif ttype == 'identity':
        pass

    else:
        assert False, ttype

    return X.reshape(-1, w*w*3)


def get_data(args):
    """
    Load samples from a dataset and apply a synthetic transformation to half of the data ("A")
    :param args: Relevant options are:
      dataset: Name of the dataset to be loaded
      n_train: Number of training images
      n_test: Number of test images
      resolution: Images will be resized to [resolution x resolution]
      pairing: 'paired' = supervised - X_A[i] = T(X_B[i])
               'matching' = The same original images are used for X_A and X_B, but in different random order
               'nonmatching' = X_A and X_B are disjoint sets (i.e. split the dataset to two parts)
               'few-matches' = Only 1/8 of the images in X_A and X_B match
      a_transform: The synthetic transformation applied to X_A (see function true_transform)
    :return: X_A, X_B, X_A_test, X_B_test
    """
    if args.dataset == 'celeba':
        train_x = get_celeba_data(num_samples=args.n_train, w=args.resolution)
        test_x = get_celeba_data('test', num_samples=args.n_test, w=args.resolution, shuffle=False)

    elif args.dataset == 'ffhq':
        train_x = get_ffhq_data(num_samples=args.n_train, w=args.resolution)
        test_x = get_ffhq_data('test', num_samples=args.n_test, w=args.resolution, shuffle=False)

    elif args.dataset == 'shoes':
        train_x = get_shoes_data(num_samples=args.n_train, w=args.resolution)
        test_x = get_shoes_data('test', num_samples=args.n_test, w=args.resolution, shuffle=False)

    n_train = train_x.shape[0]
    if args.pairing == 'nonmatching':
        X_A = train_x[:n_train//2]
        X_B = train_x[n_train//2:]

    elif args.pairing == 'few-matches':
        n_matches = n_train//8
        if (n_train-n_matches) % 2 == 1:
            n_matches += 1
        print('Inserting {}/{} matching pairs...'.format(n_matches, n_train))
        n_per_part = (n_train-n_matches) // 2
        X_A = train_x[:(n_per_part+n_matches)].copy()
        X_B = train_x[n_per_part:]

    else:
        X_A = train_x
        X_B = train_x.copy()
        if not args.pairing == 'paired':
            np.random.shuffle(X_B)

    X_A = true_transform(X_A, ttype=args.a_transform, w=args.resolution)

    X_B_test = test_x.copy()
    X_A_test = true_transform(test_x, ttype=args.a_transform, w=args.resolution)

    return X_A, X_B, X_A_test, X_B_test

