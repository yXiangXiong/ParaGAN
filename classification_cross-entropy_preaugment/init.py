import torch
import itertools
import random
import torch.nn as nn

from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
from torch.autograd import Variable
from torchvision import models

from networks import Generator, Discriminator, Classifier, define_pretrained_model


def weights_init_normal(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class HingeLoss(torch.nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, output, target):
        hinge_loss = 1 - torch.mul(output, target)
        hinge_loss[hinge_loss < 0] = 0
        return torch.mean(hinge_loss)


class LrLambda():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class Modules():
    def __init__(self, input_nc, output_nc, num_classes, model_name):
        self.netC = define_pretrained_model(model_name, num_classes)

    def init_modules(self, use_cuda, gpu_ids=[]):
        if use_cuda:
            self.netC.cuda(gpu_ids[0])
            self.netC = nn.DataParallel(self.netC, gpu_ids)


class Losses():
    def __init__(self):
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.criterion_classify = nn.BCELoss()
        self.criterion_Hinge = HingeLoss()
        self.criterion_ce = nn.CrossEntropyLoss()


class Optimizers():
    def __init__(self, module, learning_rate):
        self.optimizer_C = Adam(module.netC.parameters(), lr=learning_rate, betas=(0.5, 0.999))


class LrShedulers():
    def __init__(self, optimizer, n_epochs, epoch, decay_epoch):
        self.lr_scheduler_C = LambdaLR(optimizer.optimizer_C, lr_lambda=LrLambda(n_epochs, epoch, decay_epoch).step)


class MemoryAllocation():
    def __init__(self, cuda, batchSize, input_nc, output_nc, size, gpu_ids=[]):
        self.Tensor_G_D = torch.cuda.FloatTensor if cuda else torch.Tensor
        self.Tensor_C = torch.cuda.FloatTensor if cuda else torch.Tensor

        if cuda:
            # inputs memory allocation
            self.input_A = self.Tensor_G_D(batchSize, input_nc, size, size).cuda(gpu_ids[0])
            self.input_B = self.Tensor_G_D(batchSize, output_nc, size, size).cuda(gpu_ids[0])
            self.input_BB = self.Tensor_G_D(batchSize, output_nc, size, size).cuda(gpu_ids[0])
            self.input_OA = self.Tensor_G_D(batchSize, output_nc, size, size).cuda(gpu_ids[0])
            self.input_OB = self.Tensor_G_D(batchSize, output_nc, size, size).cuda(gpu_ids[0])

            # synthesis targets memory allocation
            self.target_real = Variable(self.Tensor_G_D(batchSize, 1).cuda(gpu_ids[0]).fill_(1.0), requires_grad=False)
            self.target_fake = Variable(self.Tensor_G_D(batchSize, 1).cuda(gpu_ids[0]).fill_(0.0), requires_grad=False)

            # classification targets memory allocation
            self.A_label = Variable(self.Tensor_C(batchSize).cuda(gpu_ids[0]).fill_(0), requires_grad=False)
            self.B_label = Variable(self.Tensor_C(batchSize).cuda(gpu_ids[0]).fill_(1), requires_grad=False)
        else:
            # inputs memory allocation
            self.input_A = self.Tensor_G_D(batchSize, input_nc, size, size)
            self.input_B = self.Tensor_G_D(batchSize, output_nc, size, size)

            # synthesis targets memory allocation
            self.target_real = Variable(self.Tensor_G_D(batchSize, 1).fill_(1.0), requires_grad=False)
            self.target_fake = Variable(self.Tensor_G_D(batchSize, 1).fill_(0.0), requires_grad=False)

            # classification targets memory allocation
            self.A_label = Variable(self.Tensor_C(batchSize).fill_(0), requires_grad=False)
            self.B_label = Variable(self.Tensor_C(batchSize).fill_(1), requires_grad=False)
            