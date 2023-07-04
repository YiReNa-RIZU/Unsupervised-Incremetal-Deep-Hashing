import torch
import os
import time
import pickle
from tqdm import tqdm
from test5 import cal_num





def cal_hashcode_num():
    def cal_all(B_data1):
        print('the kind of binary code:', cal_num(B_data1.float()))
        B_data1 = B_data1.long().to('cuda:0')
        a = torch.nonzero(B_data1 == 1)
        b = torch.nonzero(B_data1 == -1)
        
        print('the num of 1 is:', len(a))
        print('the num of -1 is:', len(b))
        
        del a,b
        
        def cal_ratio(i, B_data1):
            result = 0
            for b in B_data1:
                num_one = torch.nonzero(b == i)
                result_one = len(num_one) / len(b)
                result += result_one
            result = result / len(B_data1)
            print('the ratio of %d is:'%i, result)
            
        cal_ratio(1, B_data1)
        cal_ratio(-1, B_data1)
    
    
    
    
    root = 'checkpoints'
    up_code_root = 'B_dataset1_up.t' 
    down_code_root = 'B_dataset1_down.t' 
    
    
    B_data1_up = torch.load(os.path.join(root, up_code_root))
    B_data1_down = torch.load(os.path.join(root, down_code_root))
    B_data1 = torch.cat([B_data1_up, B_data1_down], dim = 1)
    print(B_data1.size())
    cal_all(B_data1)
    
    
    B_data1_up = torch.load(os.path.join(root, up_code_root))
    B_data1_down = torch.load(os.path.join(root, 'U_database1_down_learned.t'))
    B_data1 = torch.cat([B_data1_up, B_data1_down], dim = 1)
    print(B_data1.size())
    cal_all(B_data1)
    

# root_old = '24bits-record.pkl'
# with open(os.path.join(root, root_old), 'rb') as f:
#   B = pickle.load(f)
# B_data1 = torch.from_numpy(B['qB'])
# #B_data1 = torch.load(os.path.join(root, root_old))
# print(B_data1.size())
# cal_all(B_data1)    




