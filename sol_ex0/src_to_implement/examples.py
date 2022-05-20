import random
from cProfile import label
from math import ceil
import os
import json
from random import randint
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

class ImageGenerator:

    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.counter = 0

        # load json file for label data
        f = open(label_path)
        self.labels = list(json.load(f).items())
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.files = os.listdir('C://Users//AEPAC//Desktop//DeepLearning//sol_ex0//src_to_implement//data//exercise_data')
        self.current_epoch = 0
        self.label_len = len(self.labels)
        if self.shuffle:
            #Shuffle randomly
            np.random.shuffle(self.files)
        self.difference = len(self.files)%self.batch_size
        if self.difference != 0:
            self.files = self.files[self.batch_size-self.difference]
            #10 images -> 3
            #3,3,3,1+starting_3[:self.batch_size-difference]
        # Create batch
        self.file_chunks = list(self.chunks(self.files, self.batch_size))

    def chunks(self, lst, n):
    #"""Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def next(self):

        #epoch badana hai
        if len(self.files) >= self.label_len:
            self.counter = 0
            self.current_epoch +=1

        self.images = []
        self.image_label = []
        for file in self.file_chunks[self.counter]:
            img_path = self.file_path + file
            image = np.load(img_path)
            image = np.reshape(image,self.image_size)
            image = self.augment(image)
            self.images.append(image) #contains all images
            self.image_label.append(file[0:-4]) #contains all labels
        self.counter +=1
        return np.array(self.images), np.array(self.image_label)

    def augment(self, img):

        if self.rotation == True:
            #Where k can be 0,1,2,3 randomly
            k =  random.randint(0,3)
            img = np.rot90(img,k)

        if self.mirroring ==True:
            # mirror randomly
            img = np.fliplr(img)
        return img

    def current_epoch(self):
        return self.current_epoch

    def class_name(self, x):
        # returns class name for a specific input
        int_value = int(x)
        return self.class_dict[int_value]

    def show(self):
        images, labels = self.next()
        row = self.batch_size // 3
        col = 3
        plt.figure()
        img_num = 1
        for image, label in zip(images, labels):
            plt.subplot(row, col, img_num)
            plt.title(self.class_name(label))
            plt.imshow(image)
            img_num += 1

        plt.tight_layout()
        plt.show()

    if __name__ == '__main__':
        gen = ImageGenerator('exercise_data/', 'Labels.json', 12, [50, 50, 3], rotation=False, mirroring=False,
                             shuffle=False)
        gen.show()
c = ImageGenerator(pass)
c.show()



