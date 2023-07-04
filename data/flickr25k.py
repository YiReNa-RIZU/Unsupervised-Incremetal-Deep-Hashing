import torch
import numpy as np
import os

from PIL import Image, ImageFile
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from data.transform import train_transform, query_transform

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_data(root, num_query, num_train, batch_size, num_workers):
    """
    Loading nus-wide dataset.

    Args:
        root(str): Path of image files.
        num_query(int): Number of query data.
        num_train(int): Number of training data.
        batch_size(int): Batch size.
        num_workers(int): Number of loading data threads.

    Returns
        query_dataloader, train_dataloader, retrieval_dataloader (torch.evaluate.data.DataLoader): Data loader.
    """
    
    '''
    Flickr25k.init(root, num_query, num_train)
    query_dataset = Flickr25k(root, 'query', query_transform())
    train_dataset = Flickr25k(root, 'train', train_transform())
    retrieval_dataset = Flickr25k(root, 'retrieval', query_transform())

    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    '''
    
    query_dataset = Flickr25k(
        root,
        'test_img.txt',
        'test_label.txt',
        transform=query_transform(),
    )

    train_dataset = Flickr25k(
        root,
        'database_img.txt',
        'database_label.txt',
        transform=train_transform(),
        train=True,
        num_train=num_train,
    )

    retrieval_dataset = Flickr25k(
        root,
        'database_img.txt',
        'database_label.txt',
        transform=query_transform(),
    )
    
    
    query_dataloader = DataLoader(
        query_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
    )
    retrieval_dataloader = DataLoader(
        retrieval_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
    )

    return query_dataloader, train_dataloader, retrieval_dataloader





class Flickr25k(Dataset):
    """
    Flickr25k

    Args
        root(str): Path of image files.
        img_txt(str): Path of txt file containing image file name.
        label_txt(str): Path of txt file containing image label.
        transform(callable, optional): Transform images.
        train(bool, optional): Return training dataset.
        num_train(int, optional): Number of training data.
    """
    def __init__(self, root, img_txt, label_txt, transform=None, train=None, num_train=None):
        self.root = root
        self.transform = transform

        img_txt_path = os.path.join(root, img_txt)
        label_txt_path = os.path.join(root, label_txt)

        # Read files
        with open(img_txt_path, 'r') as f:
            self.data = np.array([i.strip() for i in f])
        self.targets = np.loadtxt(label_txt_path, dtype=np.float32)

        # Sample training dataset
        if train is True:
            perm_index = np.random.permutation(len(self.data))[:num_train]
            self.data = self.data[perm_index]
            self.targets = self.targets[perm_index]

    def __getitem__(self, index, mode='norm'):
        img = Image.open(os.path.join(self.root, 'images', self.data[index])).convert('RGB')
        if self.transform is not None and mode == 'norm':
            img = self.transform(img)
        elif mode == 'original':
            original_transforms = transforms.Compose([transforms.Resize([224,224]) ,transforms.ToTensor()])
            img = original_transforms(img)
        else:
            raise ValueError('img transform mode error')

        return img, self.targets[index], index

    def __len__(self):
        return len(self.data)

    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()



'''
class Flickr25k(Dataset):
    """
    Flicker 25k dataset.

    Args
        root(str): Path of dataset.
        mode(str, 'train', 'query', 'retrieval'): Mode of dataset.
        transform(callable, optional): Transform images.
    """
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.transform = transform

        if mode == 'train':
            self.data = Flickr25k.TRAIN_DATA
            self.targets = Flickr25k.TRAIN_TARGETS
        elif mode == 'query':
            self.data = Flickr25k.QUERY_DATA
            self.targets = Flickr25k.QUERY_TARGETS
        elif mode == 'retrieval':
            self.data = Flickr25k.RETRIEVAL_DATA
            self.targets = Flickr25k.RETRIEVAL_TARGETS
        else:
            raise ValueError(r'Invalid arguments: mode, can\'t load dataset!')

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root, 'images', self.data[index])).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[index], index

    def __len__(self):
        return self.data.shape[0]

    def get_onehot_targets(self):
        return torch.from_numpy(self.targets).float()

    @staticmethod
    def init(root, num_query, num_train):
        """
        Initialize dataset

        Args
            root(str): Path of image files.
            num_query(int): Number of query data.
            num_train(int): Number of training data.
        """
        # Load dataset
        database_img_path = os.path.join(root, 'database_img.txt')
        database_target_path = os.path.join(root, 'database_label.txt')
        
        test_img_path = os.path.join(root, 'test_img.txt')
        test_target_path = os.path.join(root, 'test_label.txt')

        # Read files
        with open(database_img_path, 'r') as f:
            data = np.array([i.strip() for i in f])
        targets = np.loadtxt(database_target_path, dtype=np.int64)

        with open(test_img_path, 'r') as f:
            test_data = np.array([i.strip() for i in f])
        test_targets = np.loadtxt(test_target_path, dtype=np.int64)
        
        # Split dataset
        if num_query > test_data.shape[0]:
            raise ValueError('number of test is out of bound')
        else:
            test_perm_index = np.random.permutation(test_data.shape[0])
            query_index = test_perm_index[:num_query]
           
        perm_index = np.random.permutation(data.shape[0])
        train_index = perm_index[:num_train]
        retrieval_index = perm_index

        Flickr25k.QUERY_DATA = test_data[query_index]
        Flickr25k.QUERY_TARGETS = test_targets[query_index, :]

        Flickr25k.TRAIN_DATA = data[train_index]
        Flickr25k.TRAIN_TARGETS = targets[train_index, :]

        Flickr25k.RETRIEVAL_DATA = data[retrieval_index]
        Flickr25k.RETRIEVAL_TARGETS = targets[retrieval_index, :]
'''