# process original pickle data to image
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def plot(image):
    plt.imshow(image)
    plt.show()

def show(image):
    img = Image.fromarray(image)
    img.show()

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def save_image(data,index=0,root='/home/lyj/cs231n-homework/data/cifar10/test'):
    images = data.get('data')
    images = np.reshape(images,(10000,32,32,3),order='F').transpose(0,2,1,3)
    labels = data.get('labels')
    for i in range(10000):
        img = Image.fromarray(images[i])
        path = os.path.join(root,f'{labels[i]}_{index}_{i}.png')
        img.save(path)

    del data
    # image_num = images.shape[0]


def make_train_data():
    data1 = unpickle('/home/lyj/cs231n-homework/data/cifar-10-batches-py/data_batch_1')
    data2 = unpickle('/home/lyj/cs231n-homework/data/cifar-10-batches-py/data_batch_2')
    data3 = unpickle('/home/lyj/cs231n-homework/data/cifar-10-batches-py/data_batch_3')
    data4 = unpickle('/home/lyj/cs231n-homework/data/cifar-10-batches-py/data_batch_4')
    data5 = unpickle('/home/lyj/cs231n-homework/data/cifar-10-batches-py/data_batch_5')
    data = [data1,data2,data3,data4,data5]
    del data1,data2,data3,data4,data5
    data  = enumerate(data)
    for index,i in data:
        save_image(i,index)

def get_name():
    names = unpickle('/home/lyj/cs231n-homework/data/cifar-10-batches-py/batches.meta')
    print(names)

def make_test_data():
    data = unpickle('/home/lyj/cs231n-homework/data/cifar-10-batches-py/test_batch')
    save_image(data)

make_test_data()