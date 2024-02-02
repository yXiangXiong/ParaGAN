import glob
import random
import os

import torchvision.transforms as transforms
import torchvision.datasets as datasets

from PIL import Image
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/X' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/Y' % mode) + '/*.*'))

    def __getitem__(self, index):
        A_path = self.files_A[index % len(self.files_A)] # cyclegan
        item_A = self.transform(Image.open(A_path).convert('RGB'))

        B_path = self.files_B[index % len(self.files_B)] # cyclegan
        item_B = self.transform(Image.open(B_path).convert('RGB'))

        return {'X': item_A, 'Y': item_B, 'X_path': A_path, 'Y_path': B_path}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class TrainLoader():
    def __init__(self, size, dataroot, batchSize, n_cpu):
        self.transform_train = [transforms.Resize((size,size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]

        self.train_loader = DataLoader(ImageDataset(dataroot, transforms_=self.transform_train, unaligned=True, mode='train'), batch_size=batchSize, shuffle=True, num_workers=n_cpu)


class ValLoader():  # special for valid the classier
    def __init__(self, size, dataroot, batchSize, n_cpu):
        val_transforms = transforms.Compose([transforms.Resize((size,size)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        self.val_set = datasets.ImageFolder(root=dataroot+'/valid', transform=val_transforms)                   
        self.val_loader = DataLoader(self.val_set, batch_size=batchSize, num_workers=n_cpu)


class TestGDLoader():
    def __init__(self, size, dataroot, batchSize, n_cpu):
        self.transform_test = [transforms.Resize((size,size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))]
    
        self.test_loader = DataLoader(ImageDataset(dataroot, transforms_=self.transform_test, mode='train'), batch_size=batchSize, shuffle=False, num_workers=n_cpu)


class TestCLoader():  # special for testing the classier
    def __init__(self, size, dataroot, batchSize, n_cpu):
        test_transforms = transforms.Compose([transforms.Resize((size,size)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        self.test_set = datasets.ImageFolder(root=dataroot+'/train', transform=test_transforms)
        self.test_loader = DataLoader(self.test_set, batch_size=batchSize, num_workers=n_cpu)