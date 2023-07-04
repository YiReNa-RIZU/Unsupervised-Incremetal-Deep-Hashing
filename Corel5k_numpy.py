import numpy as np
import os
import torch
import itertools
from PIL import Image

root = '../data-set/Corel5k'
class_num = 0
test_num = 500

train_data = np.array([])
test_data = np.array([])

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


train_data = img_list[train_index]
train_label = label[train_index].astype(int)
test_data = img_list[test_index]
test_label = label[test_index].astype(int)

print(train_data[0].dtype)

img = train_data[0]
img = Image.open(os.path.join('../data-set/Corel5k', img))
img.show()


'''
def itertools_chain(a):
    return list(itertools.chain.from_iterable(a))

train_data = [list(img_list)[x*per_class_num:x*per_class_num+per_train_num] for x in range(class_num)]
train_data = itertools_chain(train_data)
train_label = [list(label)[x*per_class_num:x*per_class_num+per_train_num] for x in range(class_num)]
train_label = itertools_chain(train_label)

test_data = [list(img_list)[x*per_class_num+per_train_num:(x+1)*per_class_num] for x in range(class_num)]
test_data = itertools_chain(test_data)
test_label = [list(label)[x*per_class_num+per_train_num:(x+1)*per_class_num] for x in range(class_num)]
test_label = itertools_chain(test_label)

img = Image.open(os.path.join(root, test_data[3]))
img.show()
test_label[3]
'''