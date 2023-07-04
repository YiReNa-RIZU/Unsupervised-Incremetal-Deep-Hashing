import torch
import numpy as np
from PIL import Image
import os
import sys
import pickle

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from data.transform import train_transform, query_transform, transform_make_matrix, Onehot, encode_onehot




def load_data(root, num_query, num_train, batch_size, num_workers):
    
    Corel5k.init(root, num_query)
    query_dataset = Corel5k(mode='query', transform=query_transform(), target_transform=Onehot())
    train_dataset = Corel5k(mode='train', transform=query_transform(), target_transform=None)
    retrieval_dataset = Corel5k(mode='database', transform=query_transform(), target_transform=Onehot())
    
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





class Corel5k(Dataset):
    
    @staticmethod
    def init(root, num_test):
        class_num = 0
        test_num = num_test


        file_data = os.listdir(root)
        file_data.sort(key = lambda x:int(x))
        img_list = np.array([])
        label = np.array([])
        for i, id in enumerate(file_data):
            img_id = os.listdir(os.path.join(root, id))
            img_id.sort(key = lambda x:int(x[:-5]))

            lab = np.array([i]).repeat(len(img_id))
            label = np.concatenate((label, lab), axis=0)
            
            img = [os.path.join(id, x) for x in img_id]
            img_list = np.concatenate((img_list, img), axis=0)
            class_num = i + 1


        all_num = len(img_list)
        train_num = all_num - test_num
        per_class_num = all_num // class_num
        per_train_num = train_num // class_num
        per_test_num = test_num // class_num
        


        train_index = np.arange(per_train_num)
        test_index = np.arange(per_train_num, per_class_num)

        train_index = np.tile(train_index, class_num)
        test_index = np.tile(test_index, class_num)

        inc_index = np.array([i*per_class_num for i in range(class_num)])
        train_index = train_index + inc_index.repeat(per_train_num)
        test_index = test_index + inc_index.repeat(per_test_num)


        Corel5k.train_data = img_list[train_index]
        Corel5k.train_label = label[train_index].astype(int)
        Corel5k.test_data = img_list[test_index]
        Corel5k.test_label = label[test_index].astype(int)
        Corel5k.class_num = class_num
        
    def __init__(self, mode, transform=None, target_transform=None):
        
        self.transform = transform
        self.target_transform = target_transform
        
        if mode == 'query':
            self.data = Corel5k.test_data
            self.targets = Corel5k.test_label
        else:
            self.data = Corel5k.train_data
            self.targets = Corel5k.train_label
            
        self.onehot_targets = encode_onehot(self.targets, Corel5k.class_num)
        
    def __getitem__(self, index):
        img_path, target = self.data[index], self.targets[index]
        
        img = Image.open(os.path.join('../data-set/Corel5k', img_path))
        
        if self.transform is not None:
            img = self.transform(img)
        
        if self.target_transform is not None:
            target = self.target_transform(target, num_classes=Corel5k.class_num)
        
        return img, target, index
    
    def __len__(self):
        return len(self.data)
    
    def get_onehot_targets(self):
        return torch.from_numpy(self.onehot_targets).float()
            