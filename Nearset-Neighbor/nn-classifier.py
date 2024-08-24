import numpy as np
import matplotlib.pyplot as plt
import pickle

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

class NearstNeighbor():
    def __init__(self):
        pass

    def l1_distance(self,new, old):
        return np.sum(np.abs(new-old),axis=1)

    def train(self, images, labels):
        self.images = images
        self.labels = labels

    def predict(self, images, labels):
        T = 0
        F = 0
        pred = np.ones(images.shape[0],dtype=int)
        for i in range(images.shape[0]):
            distance = self.l1_distance(images[i,:], self.images)
            index = np.argmin(distance)
            pred[i] = self.labels[index]
            if(labels[i] == pred[i]):
                T += 1
            else:
                F += 1
            print('min distance: ',distance[index],' T: ',T,'   F: ',F)
        acc = T/(T+F)
        return acc
    

data1 = unpickle('/home/lyj/data/cifar-10-batches-py/data_batch_1')
data2 = unpickle('/home/lyj/data/cifar-10-batches-py/data_batch_2')
data3 = unpickle('/home/lyj/data/cifar-10-batches-py/data_batch_3')
data4 = unpickle('/home/lyj/data/cifar-10-batches-py/data_batch_4')
data5 = unpickle('/home/lyj/data/cifar-10-batches-py/data_batch_5')
train_images = np.vstack((data1.get('data'),data2.get('data'),data3.get('data'),data4.get('data'),data5.get('data')))
train_labels = data1.get('labels')+data2.get('labels')+data3.get('labels')+data4.get('labels')+data5.get('labels')

data6 = unpickle('/home/lyj/data/cifar-10-batches-py/test_batch')
test_images = data6.get('data')
test_labels = data6.get('labels')

model = NearstNeighbor()
model.train(train_images, train_labels)
acc = model.predict(test_images, test_labels)
print(acc)