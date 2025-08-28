import random
import torch
import os
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class SeasonTransferDataset(Dataset):
    def __init__(self, opt):
        self.image_path = opt.dataroot
        self.is_train = opt.is_train
        self.d_num = opt.n_attribute
        print('Start preprocessing dataset..!')
        self.preprocess()
        print('Finished preprocessing dataset..!')

        trs = [transforms.Resize((256, 320))]
        if self.is_train:
            trs.append(transforms.RandomHorizontalFlip())
        trs.append(transforms.ToTensor())
        trs.append(transforms.Normalize((0.5,), (0.5,)))
        self.transform = transforms.Compose(trs)

        self.num_data = max(self.num)

    def preprocess(self):
        dirs = os.listdir(self.image_path)
        trainDirs = [dir for dir in dirs if 'train' in dir]
        testDirs = [dir for dir in dirs if 'test' in dir]
        assert len(trainDirs) == self.d_num
        trainDirs.sort()
        testDirs.sort()
        self.filenames = []
        self.num = []
        if self.is_train:
            for dir in trainDirs:
                filenames = glob("{}/{}/*.jpg".format(self.image_path, dir)) + glob(
                    "{}/{}/*.png".format(self.image_path, dir))
                filenames.sort()
                random.shuffle(filenames)
                self.filenames.append(filenames)
                self.num.append(len(filenames))
        else:
            for dir in testDirs:
                filenames = glob("{}/{}/*.jpg".format(self.image_path, dir)) + glob(
                    "{}/{}/*.png".format(self.image_path, dir))
                filenames.sort()
                self.filenames.append(filenames)
                self.num.append(len(filenames))
        self.labels = [[int(j == i) for j in range(self.d_num)] for i in range(self.d_num)]

    def __getitem__(self, index):
        imgs = []
        labels = []

        for d in range(self.d_num):
            index_d = index if index < self.num[d] else random.randint(0, self.num[d] - 1)
            if d == 0:
                res = index_d
            img = Image.open(self.filenames[d][index_d])
            img = self.transform(img)
            imgs.append(img)
            labels.append(torch.FloatTensor(self.labels[d]))
        return imgs, labels, self.filenames[0][res]

    def __len__(self):
        return self.num_data

