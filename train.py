import os
import torch
import argparse
import sys
import numpy as np
from torch.autograd import Variable
from tqdm import tqdm
from collections import OrderedDict
from init import Modules, Losses, Optimizers, LrShedulers, ReplayBuffer, MemoryAllocation
from datasets import TrainLoader, ValLoader
from utils import Logger, g_gan_loss_visualize, g_mse_loss_visualize, g_cycle_loss_visualize, d_loss_visualize, \
    val_loss_visualize, val_acc_visualize, c_hinge_real_loss_visualize, c_hinge_fake_loss_visualize
import random


def train_validation(module, loss, optimizer, lr_scheduler, trainloader, valloader, memory_allocation, opt):
    fake_A_buffer = ReplayBuffer()  # fakeA data buffer
    fake_B_buffer = ReplayBuffer()  # fakeB data buffer
    
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
        train_loss_G_GAN = 0.0
        train_loss_G_mse = 0.0
        train_loss_G_cycle = 0.0
        train_loss_D = 0.0
        train_loss_C_hinge_real = 0.0
        train_loss_C_hinge_fake = 0.0

        for i, batch in enumerate(trainloader.train_loader):
            sys.stdout.write('\rTrain epoch %03d/%03d batch [%04d/%04d]' % (epoch+1, opt.n_epochs, i+1, len(trainloader.train_loader)))
            # Set model input
            real_A = Variable(memory_allocation.input_A.copy_(batch['A']))
            real_B = Variable(memory_allocation.input_B.copy_(batch['B']))
            real_BB = Variable(memory_allocation.input_BB.copy_(batch['BB']))

            #----------------------------------------------------Auxiliary Classifier-------------------------------------------------------#
            # real_B
            distance_B_label = module.aux_C(real_B).squeeze(-1)
            # real_A 
            distance_A_label = module.aux_C(real_A).squeeze(-1)

            #-------------------------------------------------- Generators A2B and B2A ----------------------------------------------#
            optimizer.optimizer_G.zero_grad()

            fake_B = module.netG_A2B(real_A, distance_B_label.detach())
            pred_fake = module.netD_B(fake_B)
            loss_GAN_A2B = loss.criterion_GAN(pred_fake, memory_allocation.target_real)  # GAN loss

            fake_A = module.netG_B2A(real_B, distance_A_label.detach())
            pred_fake = module.netD_A(fake_A)
            loss_GAN_B2A = loss.criterion_GAN(pred_fake, memory_allocation.target_real)  # GAN loss
                
            output_fakeB = module.aux_C(fake_B).squeeze(-1)
            loss_mse_A2B = loss.criterion_mse(output_fakeB, distance_B_label.detach()) * opt.lambda_vertical
            output_fakeA = module.aux_C(fake_A).squeeze(-1)
            loss_mse_B2A = loss.criterion_mse(output_fakeA, distance_A_label.detach()) * opt.lambda_vertical

            recovered_A = module.netG_B2A(fake_B, distance_A_label.detach())
            loss_cycle_ABA = loss.criterion_cycle(recovered_A, real_A) * 10.0              # Cycle loss

            recovered_B = module.netG_A2B(fake_A, distance_B_label.detach())
            loss_cycle_BAB = loss.criterion_cycle(recovered_B, real_B) * 10.0             # Cycle loss

            loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_mse_A2B + loss_mse_B2A + loss_cycle_ABA + loss_cycle_BAB  # Total loss
            loss_G.backward()
                        
            optimizer.optimizer_G.step()

            train_loss_G_GAN += (loss_GAN_A2B + loss_GAN_B2A)
            train_loss_G_mse += (loss_mse_A2B + loss_mse_B2A)
            train_loss_G_cycle += (loss_cycle_ABA + loss_cycle_BAB)

            #---------------------------------------------------- Classifier C-------------------------------------------------------#
            optimizer.optimizer_C.zero_grad()

            # realA loss
            output_realA = module.netC(real_A).squeeze(-1)
            loss_C_realA = loss.criterion_hinge(output_realA, memory_allocation.A_label)

            # realB loss
            output_realB = module.netC(real_BB).squeeze(-1)
            loss_C_realB = loss.criterion_hinge(output_realB, memory_allocation.B_label)

            # fake_B loss
            output_fakeB = module.netC(fake_B.detach()).squeeze(-1)
            loss_C_fakeB = loss.criterion_hinge(output_fakeB, memory_allocation.B_label)

            # fake_A loss
            output_fakeA = module.netC(fake_A.detach()).squeeze(-1)
            loss_C_fakeA = loss.criterion_hinge(output_fakeA, memory_allocation.A_label)
            
            # Total loss              
            loss_C = (loss_C_realA + loss_C_realB) + (loss_C_fakeB + loss_C_fakeA) * opt.lambda_fake
            loss_C.backward()
            
            optimizer.optimizer_C.step()
            train_loss_C_hinge_real += (loss_C_realA + loss_C_realB)
            train_loss_C_hinge_fake += (loss_C_fakeB + loss_C_fakeA)

            #--------------------------------------------------- Discriminator A ----------------------------------------------------#
            optimizer.optimizer_D_A.zero_grad()
            
            pred_real = module.netD_A(real_A)
            loss_D_real = loss.criterion_GAN(pred_real, memory_allocation.target_real)   # Real loss

            fakeA_temp = fake_A # copy
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = module.netD_A(fake_A.detach())
            loss_D_fake = loss.criterion_GAN(pred_fake, memory_allocation.target_fake)   # Fake loss

            loss_D_A = (loss_D_real + loss_D_fake)*0.5                                   # Total loss
            loss_D_A.backward()

            optimizer.optimizer_D_A.step()
            train_loss_D += loss_D_A

            #--------------------------------------------------- Discriminator B ----------------------------------------------------#
            optimizer.optimizer_D_B.zero_grad()

            pred_real = module.netD_B(real_B)
            loss_D_real = loss.criterion_GAN(pred_real, memory_allocation.target_real)   # Real loss
            
            fakeB_temp = fake_B # copy
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = module.netD_B(fake_B.detach())
            loss_D_fake = loss.criterion_GAN(pred_fake, memory_allocation.target_fake)   # Fake loss
            
            loss_D_B = (loss_D_real + loss_D_fake)*0.5                                   # Total loss
            loss_D_B.backward()

            optimizer.optimizer_D_B.step()
            train_loss_D += loss_D_B

        # Progress train report
        train_logger.log({'loss_G_GAN': train_loss_G_GAN, 'loss_G_mse': train_loss_G_mse, 'loss_G_cycle': train_loss_G_cycle, 'loss_D': train_loss_D, 
                          'loss_C_hinge_real': train_loss_C_hinge_real, 'loss_C_hinge_fake': train_loss_C_hinge_fake},
                    images={'real_A': real_A, 'real_B': real_B, 'fake_A': fakeA_temp, 'fake_B': fakeB_temp, 'reconstruct_A': recovered_A, 'reconstruct_B': recovered_B})

        lr_scheduler.lr_scheduler_G.step()    # Update G learning rate
        lr_scheduler.lr_scheduler_D_A.step()  # Update D_A learning rate
        lr_scheduler.lr_scheduler_D_B.step()  # Update D_B learning rate
        lr_scheduler.lr_scheduler_C.step()    # Update C learning rate

        #*********************************************************** vaild the model***********************************************************#
        module.netC.eval()
        val_loss_C = 0.0
        val_correct_C = 0
        for data, target in tqdm(valloader.val_loader):
            target[target == 0] = -1
            data, target = data.cuda(opt.gpu_ids[0]), target.cuda(opt.gpu_ids[0])    # move to GPU or cpu
            target = target.to(torch.float)
            output = module.netC(data).squeeze(-1)                                    # forward pass
            loss_C = loss.criterion_hinge(output, target)                            # calculate the loss
            val_loss_C += loss_C.item()*data.size(0)                                 # update validation loss
            predict_y = torch.where(output >= 0, 1, -1)                              # output predicted class (i.e., idx)
            val_correct_C += (predict_y == target).sum().item()                      # update validation correct numbers

        # Progress validation report
        ave_val_loss_C = val_loss_C / len(valloader.val_loader.dataset)
        ave_val_acc_C = val_correct_C / len(valloader.val_loader.dataset)
        print('Validation Loss: {:.4f} \tValidation Accuracy: {:.4f}'.format(ave_val_loss_C, ave_val_acc_C))
        val_loss_C_list.append(ave_val_loss_C)
        val_acc_C_list.append(ave_val_acc_C)
        
        #**************************************************** show epoch loss and accuracy ****************************************************#
        g_gan_loss_visualize(epoch+1, train_logger.G_gan_loss, result_monitor_path)
        g_mse_loss_visualize(epoch+1, train_logger.G_mse_loss, result_monitor_path)
        g_cycle_loss_visualize(epoch+1, train_logger.G_cycle_loss, result_monitor_path)
        d_loss_visualize(epoch+1, train_logger.D_loss, result_monitor_path)
        c_hinge_real_loss_visualize(epoch+1, train_logger.C_hinge_real_loss, result_monitor_path)
        c_hinge_fake_loss_visualize(epoch+1, train_logger.C_hinge_fake_loss, result_monitor_path)
        val_loss_visualize(epoch+1, val_loss_C_list, result_monitor_path)
        val_acc_visualize(epoch+1, val_acc_C_list, result_monitor_path)

        #*********************************************************** save the module***********************************************************#
        if ave_val_loss_C < val_loss_C_min: # save model if validation loss has decreased
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...\n'.format(val_loss_C_min, ave_val_loss_C))
            torch.save(module.netG_A2B.state_dict(), checkpoints_path + '/best_netG_A2B.pth')
            torch.save(module.netG_B2A.state_dict(), checkpoints_path + '/best_netG_B2A.pth')
            torch.save(module.netC.state_dict(), checkpoints_path + '/best_netC.pth')
            val_loss_C_min = ave_val_loss_C

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs of training')
    parser.add_argument('--train_batchSize', type=int, default=4, help='size of the batches')
    parser.add_argument('--val_batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='./augmented_covid', help='root directory of the dataset')
    parser.add_argument('--G_lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--D_lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--C_lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--lambda_fake', type=float, default=1.0, help='initial learning rate')
    parser.add_argument('--lambda_vertical', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--decay_epoch', type=int, default=25, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--size', type=int, default=224, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--dataset_name', default='ultrasound2-3copy', type=str, help='Choose the dataset name for save path')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of classes for classification')
    parser.add_argument('--gpu_ids', type=str, default='0, 1')
    parser.add_argument('--classifier', type=str, default='./pretrained_covid/convnext_tiny_best_netC.pth', help='auxiliary classifier checkpoint file')
    parser.add_argument('--project_name', default='cyclegan_convnext_tiny', type=str, help='Choose the project name for save path')
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

    state_dict = torch.load(opt.classifier) # load pretrained aux_classifier
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module' not in k:
            k = 'module.'+k
        else:
            k = k.replace('features.module.', 'module.features.')
        new_state_dict[k]=v
    if opt.model_name == 'mnasnet1_0':
        module.aux_C.load_state_dict(state_dict)
    else:
        module.aux_C.load_state_dict(new_state_dict)
    module.aux_C.eval()
    
    loss = Losses()
    optimizer = Optimizers(module, opt.G_lr, opt.D_lr, opt.C_lr)
    lr_scheduler = LrShedulers(optimizer, opt.n_epochs, opt.epoch, opt.decay_epoch)
    trainloader = TrainLoader(opt.size, opt.dataroot, opt.train_batchSize, opt.n_cpu)
    valloader = ValLoader(opt.size, opt.dataroot, opt.val_batchSize, opt.n_cpu)
    memory_allocation = MemoryAllocation(opt.cuda, opt.train_batchSize, opt.input_nc, opt.output_nc, opt.size, opt.gpu_ids)

    train_validation(module, loss, optimizer, lr_scheduler, trainloader, valloader, memory_allocation, opt)