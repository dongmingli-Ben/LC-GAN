import os
from typing import Dict, List
from PIL import Image
import torch
from torchvision import transforms
import pandas as pd

class Dataset:

    def __init__(self, args, paths, transform=None):
        self.paths = paths
        self.data_dir = args.data_dir
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, 'Img', 
            'img_align_celeba', self.paths[idx])
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)
        return image


def parse_split(path: str) -> Dict[str, List[str]]:
    info = {}
    id2split = {'0': 'train', '1': 'valid', '2': 'test'}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                path, split = line.split()
                info[id2split[split]] = info.get(id2split[split], []) + [path]
    return info

def get_datasets(args):
    split_info = parse_split(os.path.join(args.data_dir, 'Eval', 'list_eval_partition.txt'))
    transform = transforms.Compose([
        transforms.CenterCrop(128),
        transforms.Resize(64),
        transforms.ToTensor(),
    ])
    train_ds = Dataset(args, split_info['train'], transform)
    valid_ds = Dataset(args, split_info['valid'], transform)
    test_ds = Dataset(args, split_info['test'], transform)
    return train_ds, valid_ds, test_ds

def merge_dict(dicts: List[dict]):
    merged = {}
    for key in dicts[0].keys():
        merged[key] = []
    for d in dicts:
        for key, val in d.items():
            merged[key].append(val)
    return merged

class DatasetExtract:

    def __init__(self, args, paths, transform=None):
        self.paths = paths
        self.data_dir = args.data_dir
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, 'Img', 
            'img_align_celeba', self.paths[idx])
        image = Image.open(path)
        if self.transform:
            image = self.transform(image)
        data = {
            'image': image,
            'path': path
        }
        return data

    @staticmethod
    def collate_fn(data: List[dict]):
        data = merge_dict(data)
        data['image'] = torch.stack(data['image'])
        return data

def get_extract_datasets(args):
    split_info = parse_split(os.path.join(args.data_dir, 'Eval', 'list_eval_partition.txt'))
    transform = transforms.Compose([
        transforms.CenterCrop(128),
        transforms.Resize(64),
        transforms.ToTensor(),
    ])
    train_ds = DatasetExtract(args, split_info['train'], transform)
    valid_ds = DatasetExtract(args, split_info['valid'], transform)
    test_ds = DatasetExtract(args, split_info['test'], transform)
    return train_ds, valid_ds, test_ds

class DatasetReal:

    def __init__(self, args, paths):
        self.paths = paths
        self.data_dir = args.data_dir
        self.latent_dir = args.latent_dir
        self.negative_ratio = args.negative_ratio

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = os.path.join(self.latent_dir, self.paths[idx].replace('jpg', 'pt'))
        mu, std = torch.load(path).cpu()
        z = mu + std*torch.randn_like(std)
        negative_samples = torch.randn((self.negative_ratio,) + std.shape)
        data = {
            'mu': mu,
            'std': std,
            'z': z,
            'prior': negative_samples
        }
        return data

    @staticmethod
    def collate_fn(data: List[dict]):
        data = merge_dict(data)
        data['mu'] = torch.stack(data['mu'])
        data['std'] = torch.stack(data['std'])
        data['z'] = torch.stack(data['z'])
        data['prior'] = torch.cat(data['prior'])
        data['samples'] = torch.cat([data['z'], data['prior']])
        data['labels'] = torch.cat([torch.ones(data['z'].size(0)),
                                   torch.zeros(data['prior'].size(0))])
        return data

def get_real_datasets(args):
    split_info = parse_split(os.path.join(args.data_dir, 'Eval', 'list_eval_partition.txt'))
    train_ds = DatasetReal(args, split_info['train'])
    valid_ds = DatasetReal(args, split_info['valid'])
    test_ds = DatasetReal(args, split_info['test'])
    return train_ds, valid_ds, test_ds

class DatasetAttr:

    def __init__(self, args, data):
        self.data = data
        self.data_dir = args.data_dir
        self.latent_dir = args.latent_dir

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        info = self.data.iloc[idx]
        path = os.path.join(self.latent_dir, info['path'].replace('jpg', 'pt'))
        labels = torch.tensor(info.iloc[2:].to_numpy().astype(int))
        mu, std = torch.load(path).cpu()
        z = mu + std*torch.randn_like(std)
        data = {
            'mu': mu,
            'std': std,
            'z': z,
            'attr_labels': labels
        }
        return data

    @staticmethod
    def collate_fn(data: List[dict]):
        data = merge_dict(data)
        data['mu'] = torch.stack(data['mu'])
        data['std'] = torch.stack(data['std'])
        data['z'] = torch.stack(data['z'])
        data['attr_labels'] = torch.stack(data['attr_labels'])
        return data

def get_attr_datasets(args):
    split_info = pd.read_csv(os.path.join(args.data_dir, 'Eval', 'list_eval_partition.txt'), sep=' ', header=None)
    split_info.columns = ['path', 'split']
    attr_info = pd.read_csv(os.path.join(args.data_dir, 'anno', args.annotation_name)).applymap(lambda x: 0 if x == -1 else x)
    attr_labels = ['Black_Hair', 'Blond_Hair', 'Eyeglasses', 'Male', 'Mustache', 'Smiling', 'Young']
    attr_info = attr_info[['path'] + attr_labels]
    info = pd.merge(split_info, attr_info, on='path')
    train_ds = DatasetAttr(args, info[info['split'] == 0].reset_index(drop=True))
    valid_ds = DatasetAttr(args, info[info['split'] == 1].reset_index(drop=True))
    test_ds = DatasetAttr(args, info[info['split'] == 2].reset_index(drop=True))
    return train_ds, valid_ds, test_ds, len(attr_labels)