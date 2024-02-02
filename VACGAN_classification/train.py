import os
import torch
import argparse
import sys
import numpy as np

from torch.autograd import Variable
from tqdm import tqdm
from collections import OrderedDict
from init import Modules, Losses, Optimizers, LrShedulers, MemoryAllocation
from datasets import TrainLoader, ValLoader
from utils import Logger, g_gan_loss_visualize, g_mse_loss_visualize, g_cycle_loss_visualize, d_loss_visualize, \
    val_loss_visualize, val_acc_visualize, c_hinge_real_loss_visualize, c_hinge_fake_loss_visualize, g_identity_loss_visualize
import random


def train_validation(module, loss, optimizer, lr_scheduler, trainloader, valloader, memory_allocation, opt):
    
    result_train_path = 'results/{}/{}'.format(opt.dataset_name, opt.project_name)+'/train'     # Save train results
    if not os.path.exists(result_train_path):
        os.makedirs(result_train_path)

    result_monitor_path = 'results/{}/{}'.format(opt.dataset_name, opt.project_name)+'/monitor' # Save acc and loss figures
    if not os.path.exists(result_monitor_path):
        os.makedirs(result_monitor_path)
    
    checkpoints_path = 'checkpoints/{}/{}'.format(opt.dataset_name, opt.project_name)           # Save modules checkpoints
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)

    train_logger = Logger(opt.n_epochs, len(trainloader.train_loader), result_train_path)       # train figures and loss save

    val_loss_C_min = np.Inf  # track change in minimum C validation loss
    val_loss_C_list = []
    val_acc_C_list = []
    for epoch in range(opt.epoch, opt.n_epochs):  # 0~99
        #*********************************************************** train the model ***********************************************************#
        train_loss_G = 0.0
        train_loss_D = 0.0
        train_loss_C_hinge_real = 0.0
        train_loss_C_hinge_fake = 0.0
        for i, batch in enumerate(trainloader.train_loader):
            sys.stdout.write('\rTrain epoch %03d/%03d batch [%04d/%04d]' % (epoch+1, opt.n_epochs, i+1, len(trainloader.train_loader)))
            # Set model input
            real_A = Variable(memory_allocation.input_A.copy_(batch['A']))
            real_B = Variable(memory_allocation.input_B.copy_(batch['B']))

            #-------------------------------------------------- Generator  ----------------------------------------------#
            optimizer.optimizer_G.zero_grad()
            # Sample noise and labels as generator input
            z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1, (opt.train_batchSize, opt.latent_dim))).cuda(opt.gpu_ids[0]))
            gen_labels = Variable(torch.cuda.LongTensor(np.random.randint(0, opt.num_classes, opt.train_batchSize)).cuda(opt.gpu_ids[0]))

            # Generate a batch of images
            gen_imgs = module.netG(z, gen_labels)

            # Loss measures generator's ability to fool the discriminator
            validity = module.netD(gen_imgs)
            pred_label = module.auxC(gen_imgs)
            
            g_loss = 0.5 * (loss.adversarial_loss(validity, memory_allocation.valid) + loss.auxiliary_loss(pred_label, gen_labels))
            g_loss.backward()

            optimizer.optimizer_G.step()
            train_loss_G += g_loss

            #--------------------------------------------------- Discriminator ----------------------------------------------------#
            optimizer.optimizer_D.zero_grad()
            # Loss for real images
            realA_pred  = module.netD(real_A)
            d_realA_loss = loss.adversarial_loss(realA_pred, memory_allocation.valid)

            realB_pred = module.netD(real_B)
            d_realB_loss = loss.adversarial_loss(realB_pred, memory_allocation.valid)

            # Loss for fake images
            fake_pred = module.netD(gen_imgs.detach())
            d_fake_loss = loss.adversarial_loss(fake_pred, memory_allocation.fake)

            # Total discriminator loss
            d_loss = (d_realA_loss + d_realB_loss + d_fake_loss) / 2
            d_loss.backward()

            optimizer.optimizer_D.step()
            train_loss_D += d_loss

            #---------------------------------------------------- Auxiliary C-------------------------------------------------------#
            optimizer.optimizer_auxC.zero_grad()
            realA_aux = module.auxC(real_A)
            aux_realA_loss = loss.auxiliary_loss(realA_aux, memory_allocation.A_label)

            realB_aux = module.auxC(real_B)
            aux_realB_loss = loss.auxiliary_loss(realB_aux, memory_allocation.B_label)
            
            fake_aux = module.auxC(gen_imgs.detach())
            aux_fake_loss = loss.auxiliary_loss(fake_aux, gen_labels)
            
            # Total auxiliary classifier loss
            loss_auxC = (aux_realA_loss + aux_realB_loss + aux_fake_loss) / 2
            loss_auxC.backward()
            
            optimizer.optimizer_auxC.step()
            
            #---------------------------------------------------- Classifier C-------------------------------------------------------#
            optimizer.optimizer_C.zero_grad()

            # realA loss
            output_realA = module.netC(real_A).squeeze(-1)
            loss_C_realA = loss.auxiliary_loss(output_realA, memory_allocation.A_label)

            # realB loss
            output_realB = module.netC(real_B).squeeze(-1)
            loss_C_realB = loss.auxiliary_loss(output_realB, memory_allocation.B_label)

            # fake loss
            output_fakeB = module.netC(gen_imgs.detach()).squeeze(-1)
            loss_C_fake = loss.auxiliary_loss(output_fakeB, gen_labels)
            
            # Total loss              
            loss_C = (loss_C_realA + loss_C_realB)*0.5 + loss_C_fake
            loss_C.backward()
            
            optimizer.optimizer_C.step()
            train_loss_C_hinge_real += (loss_C_realA + loss_C_realB)
            train_loss_C_hinge_fake += loss_C_fake

        # Progress train report
        train_logger.log({'loss_G': train_loss_G, 'loss_D': train_loss_D, 'loss_C_hinge_real': train_loss_C_hinge_real, 'loss_C_hinge_fake': train_loss_C_hinge_fake},
                    images={'real_A': real_A, 'real_B': real_B, 'generation': gen_imgs})

        lr_scheduler.lr_scheduler_G.step()    # Update G learning rate
        lr_scheduler.lr_scheduler_D.step()    # Update D learning rate
        lr_scheduler.lr_scheduler_C.step()    # Update C learning rate

        #*********************************************************** vaild the model***********************************************************#
        module.netC.eval()
        val_loss_C = 0.0
        val_correct_C = 0
        for data, target in tqdm(valloader.val_loader):
            data, target = data.cuda(opt.gpu_ids[0]), target.cuda(opt.gpu_ids[0])    # move to GPU or cpu
            output = module.netC(data)                    # forward pass: compute predicted outputs by passing inputs to the model
            loss_C = loss.auxiliary_loss(output, target)  # calculate the batch loss
            val_loss_C += loss_C.item()*data.size(0)      # update validation loss
            predict_y = torch.max(output, dim=1)[1]
            val_correct_C += (predict_y == target).sum().item()   # update validation correct numbers

        # Progress validation report
        ave_val_loss_C = val_loss_C / len(valloader.val_loader.dataset)
        ave_val_acc_C = val_correct_C / len(valloader.val_loader.dataset)
        print('Validation Loss: {:.4f} \tValidation Accuracy: {:.4f}'.format(ave_val_loss_C, ave_val_acc_C))
        val_loss_C_list.append(ave_val_loss_C)
        val_acc_C_list.append(ave_val_acc_C)
        
        #**************************************************** show epoch loss and accuracy ****************************************************#
        g_gan_loss_visualize(epoch+1, train_logger.G_loss, result_monitor_path)
        d_loss_visualize(epoch+1, train_logger.D_loss, result_monitor_path)
        val_loss_visualize(epoch+1, val_loss_C_list, result_monitor_path)
        val_acc_visualize(epoch+1, val_acc_C_list, result_monitor_path)
        c_hinge_real_loss_visualize(epoch+1, train_logger.C_hinge_real_loss, result_monitor_path)
        c_hinge_fake_loss_visualize(epoch+1, train_logger.C_hinge_fake_loss, result_monitor_path)

        #*********************************************************** save the module***********************************************************#
        if ave_val_loss_C < val_loss_C_min: # save model if validation loss has decreased
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...\n'.format(val_loss_C_min, ave_val_loss_C))
            torch.save(module.netG.state_dict(), checkpoints_path + '/best_netG.pth')
            torch.save(module.netC.state_dict(), checkpoints_path + '/best_netC.pth')
            val_loss_C_min = ave_val_loss_C

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
    parser.add_argument('--train_batchSize', type=int, default=3, help='size of the batches')
    parser.add_argument('--val_batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='/mnt/evo1/xiangyu/data/aaai2024/augmented_covid', help='root directory of the dataset')
    parser.add_argument('--G_lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--D_lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--C_lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument('--decay_epoch', type=int, default=25, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=224, help='size of the data crop (squared assumed)')
    parser.add_argument('--channels', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--dataset_name', default='covid6', type=str, help='Choose the dataset name for save path')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of classes for classification')
    parser.add_argument('--gpu_ids', type=str, default='2, 3')
    parser.add_argument('--project_name', default='vacgan_convnext_tiny', type=str, help='Choose the project name for save path')
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

    module = Modules(opt.size, opt.channels, opt.num_classes, opt.latent_dim, opt.model_name)
    module.init_modules(opt.cuda, opt.gpu_ids)

    loss = Losses()
    optimizer = Optimizers(module, opt.G_lr, opt.D_lr, opt.C_lr)
    lr_scheduler = LrShedulers(optimizer, opt.n_epochs, opt.epoch, opt.decay_epoch)
    trainloader = TrainLoader(opt.size, opt.dataroot, opt.train_batchSize, opt.n_cpu)
    valloader = ValLoader(opt.size, opt.dataroot, opt.val_batchSize, opt.n_cpu)
    memory_allocation = MemoryAllocation(opt.cuda, opt.train_batchSize, opt.channels, opt.size, opt.gpu_ids)

    train_validation(module, loss, optimizer, lr_scheduler, trainloader, valloader, memory_allocation, opt)