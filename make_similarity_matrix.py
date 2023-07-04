import sys,os
import torch
import torchvision.transforms as transforms
import time
from PIL import Image 
from data.data_loader import load_data
from SCAN import read_model
import numpy as np



def make_time(fn):
    
    def cal_time(*args):
        start_time = time.time()
        result = fn(*args)
        print('The time of making similarity matirx: ',time.time()-start_time)
        return result
        
    return cal_time
        


def encode_onehot(labels, num_classes=10):
    """
    one-hot labels

    Args:
        labels (numpy.ndarray): labels.
        num_classes (int): Number of classes.

    """
    onehot_labels = np.zeros((len(labels), num_classes))

    for i in range(len(labels)):
        onehot_labels[i, labels[i].astype('int32')] = 1

    return onehot_labels

'''
query_dataloader, _, retrieval_dataloader, make_matrix_dataloader = load_data(
    'cifar-10',
    '../data-set',
    1000,
    2000,
    64,
    0,
)
'''

@make_time
def make_similarity_maxtrix(mode, dataloader, match):

    m = read_model.Read_model('scan','SCAN').to('cuda:0')
    #print(m)
    #match = [(0, 5), (1, 9), (2, 3), (3, 2), (4, 0), (5, 1), (6, 6), (7, 7), (8, 8), (9, 4)]
    
    
    with torch.no_grad():
        
        virtual_label = np.array([])
        for img,lab,index in dataloader:
            img = img.to('cuda:0')
            output = m(img)
            #print(output)
            #print(output[0].argmax(1))
            a = output[0].argmax(1)
            reordered_preds = torch.zeros(lab.size(0))
            for t in match:
                reordered_preds[a == t[0]] = t[1]
                
            virtual_label = np.concatenate((virtual_label,reordered_preds))
            
        virtual_label_onehot = encode_onehot(virtual_label)
        virtual_label_onehot = torch.from_numpy(virtual_label_onehot)
        print('The size of similarity matirx: ', virtual_label_onehot.shape)
        #print(virtual_label_onehot)
        #print(virtual_label)
    return virtual_label_onehot
    
    


