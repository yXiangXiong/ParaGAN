import torch
import itertools
import random
import torch.nn as nn
import numpy as np

from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
from torch.autograd import Variable
from torchvision import models

from networks import Generator, Discriminator, define_pretrained_model


def weights_init_normal(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class LrLambda():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


class Modules():
    def __init__(self, img_size, channels, num_classes, latent_dim, model_name):
        self.netG = Generator(img_size, channels, latent_dim, num_classes)
        self.netD = Discriminator(img_size, channels, num_classes)
        self.netC = define_pretrained_model(model_name, num_classes)

    def init_modules(self, use_cuda, gpu_ids=[]):
        if use_cuda:
            self.netG.cuda(gpu_ids[0])
            self.netG = nn.DataParallel(self.netG, gpu_ids)
            self.netD.cuda(gpu_ids[0])
            self.netD = nn.DataParallel(self.netD, gpu_ids)
            self.netC.cuda(gpu_ids[0])
            self.netC = nn.DataParallel(self.netC, gpu_ids)

        self.netG.apply(weights_init_normal)
        self.netD.apply(weights_init_normal)


class Losses():
    def __init__(self):
        self.adversarial_loss = nn.BCELoss()
        self.auxiliary_loss = nn.CrossEntropyLoss()


class Optimizers():
    def __init__(self, module, G_learning_rate, D_learning_rate, C_learning_rate):
        self.optimizer_G = Adam(module.netG.parameters(), lr=G_learning_rate, betas=(0.5, 0.999))
        self.optimizer_D = Adam(module.netD.parameters(), lr=D_learning_rate, betas=(0.5, 0.999))
        self.optimizer_C = Adam(module.netC.parameters(), lr=C_learning_rate, betas=(0.5, 0.999))


class LrShedulers():
    def __init__(self, optimizer, n_epochs, epoch, decay_epoch):
        self.lr_scheduler_G = LambdaLR(optimizer.optimizer_G, lr_lambda=LrLambda(n_epochs, epoch, decay_epoch).step)
        self.lr_scheduler_D = LambdaLR(optimizer.optimizer_D, lr_lambda=LrLambda(n_epochs, epoch, decay_epoch).step)
        self.lr_scheduler_C = LambdaLR(optimizer.optimizer_C, lr_lambda=LrLambda(n_epochs, epoch, decay_epoch).step)


class MemoryAllocation():
    def __init__(self, cuda, batchSize, channels, size, gpu_ids=[]):
        self.Tensor_G_D = torch.cuda.FloatTensor if cuda else torch.Tensor
        self.Tensor_Long = torch.cuda.LongTensor if cuda else torch.Tensor

        if cuda:
            # inputs memory allocation
            self.input_A = self.Tensor_G_D(batchSize, channels, size, size).cuda(gpu_ids[0])
            self.input_B = self.Tensor_G_D(batchSize, channels, size, size).cuda(gpu_ids[0])

            # synthesis targets memory allocation
            self.valid = Variable(self.Tensor_G_D(batchSize, 1).cuda(gpu_ids[0]).fill_(1.0), requires_grad=False)
            self.fake = Variable(self.Tensor_G_D(batchSize, 1).cuda(gpu_ids[0]).fill_(0.0), requires_grad=False)

            # classification targets memory allocation
            self.A_label = Variable(self.Tensor_Long(batchSize).cuda(gpu_ids[0]).fill_(0.0), requires_grad=False)
            self.B_label = Variable(self.Tensor_Long(batchSize).cuda(gpu_ids[0]).fill_(1.0), requires_grad=False)
            
        else:
            # inputs memory allocation
            self.input_A = self.Tensor_G_D(batchSize, channels, size, size)
            self.input_B = self.Tensor_G_D(batchSize, channels, size, size)

            # synthesis targets memory allocation
            self.target_real = Variable(self.Tensor_G_D(batchSize, 1).fill_(1.0), requires_grad=False)
            self.target_fake = Variable(self.Tensor_G_D(batchSize, 1).fill_(0.0), requires_grad=False)

            # classification targets memory allocation
            self.A_label = Variable(self.Tensor_C(batchSize).fill_(0), requires_grad=False)
            self.B_label = Variable(self.Tensor_C(batchSize).fill_(1), requires_grad=False)
            