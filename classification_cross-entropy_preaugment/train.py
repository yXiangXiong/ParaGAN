import os
import torch
import argparse
import sys
import numpy as np
import random

from torch.autograd import Variable
from tqdm import tqdm

from init import Modules, Losses, Optimizers, LrShedulers, ReplayBuffer, MemoryAllocation
from datasets import TrainLoader, ValLoader
from utils import Logger, g_loss_visualize, g_identity_loss_visualize, g_gan_loss_visualize, g_cycle_loss_visualize, d_loss_visualize, c_loss_visualize, c_acc_visualize, val_loss_visualize, val_acc_visualize


def train_validation(module, loss, optimizer, lr_scheduler, trainloader, valloader, memory_allocation, opt):
    fake_A_buffer = ReplayBuffer()  # fakeA data buffer
    fake_B_buffer = ReplayBuffer()  # fakeB data buffer

    result_monitor_path = 'results/{}/{}'.format(opt.dataset_name, opt.project_name)+'/monitor' # Save acc and loss figures
    if not os.path.exists(result_monitor_path):
        os.makedirs(result_monitor_path)
    
    checkpoints_path = 'checkpoints/{}/{}'.format(opt.dataset_name, opt.project_name)           # Save modules checkpoints
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    val_loss_C_min = np.Inf  # track change in minimum C validation loss
    val_loss_C_list = []
    val_acc_C_list = []
    for epoch in range(opt.epoch, opt.n_epochs):  # 0~99
        #*********************************************************** train the model ***********************************************************#
        for i, batch in enumerate(trainloader.train_loader):
            sys.stdout.write('\rTrain epoch %03d/%03d batch [%04d/%04d]' % (epoch+1, opt.n_epochs, i+1, len(trainloader.train_loader)))
            # Set model input
            real_A = Variable(memory_allocation.input_A.copy_(batch['X']))
            real_B = Variable(memory_allocation.input_B.copy_(batch['Y']))

            #---------------------------------------------------- Classifier C-------------------------------------------------------#
            optimizer.optimizer_C.zero_grad()

            # realA loss
            output_realA = module.netC(real_A).squeeze(-1)
            loss_C_realA = loss.criterion_ce(output_realA, memory_allocation.A_label.long())

            # realB loss
            output_realB = module.netC(real_B).squeeze(-1)
            loss_C_realB = loss.criterion_ce(output_realB, memory_allocation.B_label.long())
            
            # Total loss              
            loss_C = loss_C_realA + loss_C_realB
            loss_C.backward()
            
            optimizer.optimizer_C.step()

        lr_scheduler.lr_scheduler_C.step()    # Update C learning rate

        #*********************************************************** vaild the model***********************************************************#
        module.netC.eval()
        val_loss_C = 0.0
        val_correct_C = 0
        for data, target in tqdm(valloader.val_loader):
            data, target = data.cuda(opt.gpu_ids[0]), target.cuda(opt.gpu_ids[0])    # move to GPU or cpu
            output = module.netC(data).squeeze(1)                                    # forward pass
            loss_C = loss.criterion_ce(output, target)                               # calculate the loss
            val_loss_C += loss_C.item()*data.size(0)                                 # update validation loss
            predict_y = torch.max(output, dim=1)[1]                                     # output predicted class (i.e., idx)
            val_correct_C += (predict_y == target).sum().item()                      # update validation correct numbers

        # Progress validation report
        ave_val_loss_C = val_loss_C / len(valloader.val_loader.dataset)
        ave_val_acc_C = val_correct_C / len(valloader.val_loader.dataset)
        print('Validation Loss: {:.4f} \tValidation Accuracy: {:.4f}'.format(ave_val_loss_C, ave_val_acc_C))
        val_loss_C_list.append(ave_val_loss_C)
        val_acc_C_list.append(ave_val_acc_C)
        
        #**************************************************** show epoch loss and accuracy ****************************************************#
        val_loss_visualize(epoch+1, val_loss_C_list, result_monitor_path)
        val_acc_visualize(epoch+1, val_acc_C_list, result_monitor_path)

        #*********************************************************** save the module***********************************************************#
        if ave_val_loss_C < val_loss_C_min: # save model if validation loss has decreased
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...\n'.format(val_loss_C_min, ave_val_loss_C))
            torch.save(module.netC.state_dict(), checkpoints_path + '/{}_best_netC.pth'.format(opt.model_name))
            val_loss_C_min = ave_val_loss_C


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
    parser.add_argument('--train_batchSize', type=int, default=4, help='size of the batches')
    parser.add_argument('--val_batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='/mnt/evo1/xiangyu/data/aaai2024/augmented_ultrasound', help='root directory of the dataset')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=25, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=224, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--dataset_name', default='ultrasound1', type=str, help='Choose the dataset name for save path')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of classes for classification')
    parser.add_argument('--gpu_ids', type=str, default='2, 3')
    parser.add_argument('--project_name', default='convnext_tiny', type=str, help='Choose the project name for save path')
    parser.add_argument('--model_name', default='convnext_tiny', type=str, 
        choices=['alexnet', 'vgg13', 'vgg16', 'googlenet', 'resnet18', 'resnet34', 'densenet121',
        'mnasnet1_0', 'mobilenet_v3_small', 'efficientnet_b5', 'convnext_tiny'], help='Choose the model you want train')
    opt = parser.parse_args()

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    if opt.cuda:
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)

    module = Modules(opt.input_nc, opt.output_nc, opt.num_classes, opt.model_name)
    module.init_modules(opt.cuda, opt.gpu_ids)
    loss = Losses()
    optimizer = Optimizers(module, opt.lr)
    lr_scheduler = LrShedulers(optimizer, opt.n_epochs, opt.epoch, opt.decay_epoch)
    trainloader = TrainLoader(opt.size, opt.dataroot, opt.train_batchSize, opt.n_cpu)
    valloader = ValLoader(opt.size, opt.dataroot, opt.val_batchSize, opt.n_cpu)
    memory_allocation = MemoryAllocation(opt.cuda, opt.train_batchSize, opt.input_nc, opt.output_nc, opt.size, opt.gpu_ids)

    train_validation(module, loss, optimizer, lr_scheduler, trainloader, valloader, memory_allocation, opt)