import random
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

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
        self.labels = dict(json.load(f).items())
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.files = os.listdir(str(self.file_path))
        self.epoch_count = 0
        self.label_len = len(self.labels)
        if self.shuffle:
            self.files = np.random.permutation(self.files)
        self.difference = len(self.files)%self.batch_size
        if self.difference != 0:
            self.files += self.files[:self.batch_size-self.difference]

        # Create batch
        self.file_chunks = list(self.chunks(self.files, self.batch_size))

    def chunks(self, lst, n):

        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def next(self):

        if self.counter * self.batch_size >= len(self.files):
            self.counter = 0
            self.epoch_count +=1

            if self.shuffle == True:
                self.files = os.listdir(self.file_path)
                if self.difference != 0:
                    self.files += self.files[:self.batch_size - self.difference]
                    self.shuffle = False

                # Create batch
                self.file_chunks = list(self.chunks(self.files, self.batch_size))



        self.images = []
        self.image_label = []
        for file in self.file_chunks[self.counter]:
            img_path = self.file_path + file
            image = np.load(img_path)
            image = np.resize(image, self.image_size)
            image = self.augment(image)
            self.images.append(image)
            self.image_label.append(int(file[0:-4]))
        self.counter +=1
        return np.array(self.images), np.array(self.image_label)

    def augment(self, img):

        if self.rotation == True:
            k =  random.randint(0,3)
            img = np.rot90(img,k)

        if self.mirroring ==True:
            # mirror randomly
            img = np.fliplr(img)
        return img

    def current_epoch(self):
        return self.epoch_count

    def class_name(self, x):

        int_value = int(x)
        return self.class_dict[int_value]

    def show(self):

        def show(self):
            # TODO: implement show method
            images, labels = self.next()
            rows = self.batch_size // 3
            pos = range(0, self.batch_size)
            fig = plt.figure(1, tight_layout=True)
            for i in range(self.batch_size):

                axes = fig.add_subplot(rows, 3, pos[i])
                axes.imshow(images[i])
                axes.set_title(self.class_name(self.json_data[str(labels[i])]))

            plt.show()
            return

gen = ImageGenerator('exercise_data/', 'Labels.json', 10, [50, 50, 3], rotation=False, mirroring=False,
                         shuffle=True)
gen.next()
gen.show()




