import argparse
import sys
import os
import torch
import cv2

import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from utils import plot_confusion_matrix, plot_roc_auc_curve, tensor2im

from init import Modules, MemoryAllocation, HingeLoss
from datasets import TestGDLoader, TestCLoader
from utils import FileName
from feature_show import define_model_trunc, plot_2d_features, plot_3d_features
from PIL import Image

import warnings
warnings.filterwarnings("ignore")
import random


def test_heatmap(module, dataloader, data_c_loader, memory_allocation, opt):  # synthetic images from generators
    result_test_path = 'results/{}/{}'.format(opt.dataset_name, opt.project_name) +'/test'
    A2zero_path = result_test_path + '/A2zero'
    B2zero_path = result_test_path + '/B2zero'
    A2zero_Heatmap_path = result_test_path + '/A2zero_heatmap'
    B2zero_Heatmap_path = result_test_path + '/B2zero_heatmap'

    if not os.path.exists(A2zero_path):
        os.makedirs(A2zero_path)
    if not os.path.exists(B2zero_path):
        os.makedirs(B2zero_path)
    if not os.path.exists(A2zero_Heatmap_path):
        os.makedirs(A2zero_Heatmap_path)
    if not os.path.exists(B2zero_Heatmap_path):
        os.makedirs(B2zero_Heatmap_path)

    for i, batch in enumerate(dataloader.test_loader):
        real_A = Variable(memory_allocation.input_A.copy_(batch['A']))  # Set model input realA
        real_BB = Variable(memory_allocation.input_BB.copy_(batch['BB']))  # Set model input realB   

        fake_A2zero = 0.5*(module.netG_A2B(real_A, memory_allocation.target_zero)[0].data + 1.0)        # Generate output fakeB
        fake_B2zero = 0.5*(module.netG_B2A(real_BB, memory_allocation.target_zero)[0].data + 1.0)        # Generate output fakeA

        heatmapA = abs(real_A - fake_A2zero)
        heatmapA = tensor2im(heatmapA.data)
        heatmapA = heatmapA / np.max(heatmapA)
        heatmapA = cv2.applyColorMap(np.uint8(255*heatmapA), cv2.COLORMAP_JET)
        cv2.cvtColor(heatmapA, cv2.COLOR_BGR2RGB)
        heatmapA = np.float32(heatmapA) / 255
        imgA = Image.open(batch['A_path'][0]).convert('RGB')
        imgA = np.array(imgA, dtype=np.uint8)
        imgA = cv2.resize(imgA, (224, 224))
        imgA = imgA.astype(dtype=np.float32) / 255.0
        camA = heatmapA + imgA
        camA = camA / np.max(camA)
        camA = np.uint8(255 * camA)

        heatmapB = abs(real_BB - fake_B2zero)
        heatmapB = tensor2im(heatmapB.data)
        heatmapB = heatmapB / np.max(heatmapB)
        heatmapB = cv2.applyColorMap(np.uint8(255*heatmapB), cv2.COLORMAP_JET)
        cv2.cvtColor(heatmapB, cv2.COLOR_BGR2RGB)
        heatmapB = np.float32(heatmapB) / 255
        imgB = Image.open(batch['BB_path'][0]).convert('RGB')
        imgB = np.array(imgB, dtype=np.uint8)
        imgB = cv2.resize(imgB, (224, 224))
        imgB = imgB.astype(dtype=np.float32) / 255.0
        camB = heatmapB + imgB
        camB = camB / np.max(camB)
        camB = np.uint8(255 * camB)

        A_name = FileName(batch['A_path'][0])                      # Set result A name
        BB_name = FileName(batch['BB_path'][0])                    # Set result B name
        save_image(fake_A2zero, A2zero_path + '/X2zero{}png'.format(A_name))  # Save image files fakeA
        save_image(fake_B2zero, B2zero_path + '/Y2zero{}png'.format(BB_name))  # Save image files fakeB
        colorA_pil = Image.fromarray(camA)
        colorB_pil = Image.fromarray(camB)
        colorA_pil.save(A2zero_Heatmap_path + '/X_heatmap{}png'.format(A_name))  # Save image colormap A
        colorB_pil.save(B2zero_Heatmap_path + '/Y_heatmap{}png'.format(BB_name)) # Save image colormap B

        sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader.test_loader)))


def test_gd(module, dataloader, data_c_loader, memory_allocation, opt):  # synthetic images from generators
    result_test_path = 'results/{}/{}'.format(opt.dataset_name, opt.project_name) +'/test'
    fakeA_path = result_test_path + '/fakeA'
    fakeB_path = result_test_path + '/fakeB'
    realA_path = result_test_path + '/realA'
    realB_path = result_test_path + '/realB'
    if not os.path.exists(fakeA_path):
        os.makedirs(fakeA_path)
    if not os.path.exists(fakeB_path):
        os.makedirs(fakeB_path)
    if not os.path.exists(realA_path):
        os.makedirs(realA_path)
    if not os.path.exists(realB_path):
        os.makedirs(realB_path)
    encoding_array = []  
    class_to_idx = data_c_loader.test_set.class_to_idx   
    feature_path = 'results/{}/{}'.format(opt.dataset_name, opt.project_name) +'/test/classification'
    model_trunc = define_model_trunc(opt.project_name, module.netC)  # for plotting features 
    x = []

    for i, batch in enumerate(dataloader.test_loader):
        real_A = Variable(memory_allocation.input_A.copy_(batch['A']))    # Set model input realA
        real_BB = Variable(memory_allocation.input_B.copy_(batch['B']))  # Set model input realB
                    
        # real_B
        distance_B_label = module.aux_C(real_BB).squeeze(-1)
        # real_A 
        distance_A_label = module.aux_C(real_A).squeeze(-1)

        fb = module.netG_A2B(real_A, distance_B_label.detach())
        fa = module.netG_B2A(real_BB, distance_A_label.detach())

        fake_B = 0.5*(module.netG_A2B(real_A, distance_B_label.detach())[0].data + 1.0)        # Generate output fakeB
        fake_A = 0.5*(module.netG_B2A(real_BB, distance_A_label.detach())[0].data + 1.0)        # Generate output fakeA

        for j in range(opt.batchSize):
            feature = model_trunc(real_A[j].unsqueeze(0))['semantic_feature'].squeeze().detach().cpu().numpy()
            x.append(0)
            encoding_array.append(feature)
            feature = model_trunc(real_BB[j].unsqueeze(0))['semantic_feature'].squeeze().detach().cpu().numpy()
            x.append(1)
            encoding_array.append(feature)
            feature = model_trunc(fa[j].unsqueeze(0))['semantic_feature'].squeeze().detach().cpu().numpy()
            x.append(0)
            encoding_array.append(feature)
            feature = model_trunc(fb[j].unsqueeze(0))['semantic_feature'].squeeze().detach().cpu().numpy()
            encoding_array.append(feature)
            x.append(1)

        A_name = FileName(batch['A_path'][0])                    # Set result A name
        B_name = FileName(batch['BB_path'][0])                    # Set result B name
        save_image(fake_A, fakeA_path + '/{}png'.format(B_name))  # Save image files fakeA
        save_image(fake_B, fakeB_path + '/{}png'.format(A_name))  # Save image files fakeB
        save_image(real_A, realA_path + '/{}png'.format(A_name))  # Save image files realA
        save_image(real_BB, realB_path + '/{}png'.format(B_name))  # Save image files realB

        sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader.test_loader)))
    encoding_array = np.array(encoding_array)
    testset_targets = np.array(x)
    plot_2d_features(encoding_array, class_to_idx, feature_path, testset_targets)
    plot_3d_features(encoding_array, class_to_idx, feature_path, testset_targets)
    print('The 2D and 3D features have been plotted\n')

    sys.stdout.write('\n')


def test_c(module, dataloader, opt):  # evaluate the performance of downstream classifiers
    test_loss = 0.0                                   # keep track of testing loss
    test_correct = 0                                  # keep track of testing correct numbers

    class_correct = list(0 for i in range(2))         # keep track of each category's correct numbers
    class_total = list(0 for i in range(2))           # acculmulate each category's toal numbers

    classes = dataloader.test_set.classes             # accuracy for each category & confusion matrix

    pred_cm = torch.cuda.FloatTensor([])    # for confusion matrix
    pred_cm = pred_cm.cuda(opt.gpu_ids[0])  # for confusion matrix

    y_true = []  # init for AUC and ROC
    y_prob = []  # init for AUC and ROC

    class_to_idx = dataloader.test_set.class_to_idx                  # for plotting features
    print('The classification objects and indexes: ', class_to_idx)
    encoding_array = []                                              # for plotting features
    model_trunc = define_model_trunc(opt.project_name, module.netC)  # for plotting features

    criterion = HingeLoss()          # define the cost function

    for data, target in tqdm(dataloader.test_loader):
        data, target = data.cuda(opt.gpu_ids[0]), target.cuda(opt.gpu_ids[0])    # move to GPU or cpu
        output = module.netC(data).squeeze(-1)                                  # forward pass
        loss = criterion(output, target)                     # calculate the loss
        test_loss += loss.item()*data.size(0)                # update testing loss

        predict_y = torch.where(output >= 0, 1, 0)           # output predicted class (i.e., idx)

        test_correct += (predict_y == target).sum().item()   # update validation correct numbers
        
        correct_tensor = (predict_y == target)
        for i in range(opt.batchSize):
            c = target[i]
            class_correct[c] += correct_tensor[i].item()
            class_total[c] += 1

            y_true_ = np.squeeze(target[i].data.cpu().float().numpy())
            y_true_[y_true_==0] = -1
            y_true.append(int(y_true_))
            y_prob_ = np.squeeze(output[i].data.cpu().float().numpy())
            y_prob.append(y_prob_)

           # print(data[i].unsqueeze(0).shape)
            feature = model_trunc(data[i].unsqueeze(0))['semantic_feature'].squeeze().detach().cpu().numpy()
            encoding_array.append(feature)

        pred_cm = torch.cat((pred_cm, predict_y), dim=0)

    ave_test_loss = test_loss/len(dataloader.test_loader.dataset)      # calculate average loss
    print('\nTesting Loss: {:.6f}'.format(ave_test_loss))

    ave_test_acc = test_correct/len(dataloader.test_loader.dataset)    # calculate average accuracy
    print('Testing Accuracy (Overall): {:.4f} ({}/{})'.format(ave_test_acc, test_correct, len(dataloader.test_loader.dataset)))

    for i in range(2):  #  output accuracy for each category
        if class_total[i] > 0:
            print('Testing accuracy of {}: {:.4f} ({}/{})'.format(classes[i], class_correct[i]/class_total[i], class_correct[i], class_total[i]))
        else:
            print('Testing accuracy of {}: N/A (no training examples)' % (classes[i]))

    print('\nThe Confusion Matrix is plotted and saved:')
    cMatrix = confusion_matrix(torch.tensor(dataloader.test_set.targets), pred_cm.cpu())
    print(cMatrix)
    
    result_path ='results/{}/{}'.format(opt.dataset_name, opt.project_name) +'/test/classification'
    if not os.path.exists(result_path): 
        os.makedirs(result_path)

    plot_confusion_matrix(classes, cMatrix, result_path)

    print('\nThe Classification Report is plotted below:')
    print(classification_report(torch.tensor(dataloader.test_set.targets), pred_cm.cpu()))

    auc = roc_auc_score(y_true, y_prob)
    print('The AUC (Area Under Curve) is: {:.6f}'.format(auc))
    plot_roc_auc_curve(y_true, y_prob, opt.project_name, result_path)
    print('The ROC curve have been plotted\n')
    
    feature_path = 'results/{}/{}'.format(opt.dataset_name, opt.project_name) +'/test/classification'
    if not os.path.exists(feature_path): 
        os.makedirs(feature_path)
    encoding_array = np.array(encoding_array)
    testset_targets = np.array(dataloader.test_set.targets)
    plot_2d_features(encoding_array, class_to_idx, feature_path, testset_targets)
    plot_3d_features(encoding_array, class_to_idx, feature_path, testset_targets)
    print('The 2D and 3D features have been plotted\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, choices=[1], default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='./augmented_covid', help='root directory of the dataset')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=224, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--num_classes', default=1, type=int, help='Number of classes for classification')
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--dataset_name', default='covid1-3_1.0', type=str, help='Choose the dataset name for save results')
    parser.add_argument('--generator_A2B', type=str, default='checkpoints/covid1-3_1.0/cyclegan_convnext_tiny/best_netG_A2B.pth', help='A2B generator checkpoint file')
    parser.add_argument('--generator_B2A', type=str, default='checkpoints/covid1-3_1.0/cyclegan_convnext_tiny/best_netG_B2A.pth', help='B2A generator checkpoint file')
    parser.add_argument('--project_name', default='cyclegan_convnext_tiny', type=str, help='Choose the project name for save results')
    parser.add_argument('--classifier', type=str, default='checkpoints/covid1-3_1.0/cyclegan_convnext_tiny/best_netC.pth', help='classifier checkpoint file')
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

    # Load state dicts for generators and a classifier
    module.netG_A2B.load_state_dict(torch.load(opt.generator_A2B))
    module.netG_B2A.load_state_dict(torch.load(opt.generator_B2A))
    module.netC.load_state_dict(torch.load(opt.classifier))

    # Set module networks test mode
    module.netG_A2B.eval()
    module.netG_B2A.eval()
    module.netC.eval()
    
    # test the classifier performance
    data_c_loader = TestCLoader(opt.size, opt.dataroot, opt.batchSize, opt.n_cpu)
    test_c(module, data_c_loader, opt)

    # test the cyclegan (Generator & Disciminator) performance
    data_gd_loader = TestGDLoader(opt.size, opt.dataroot, opt.batchSize, opt.n_cpu)
    memory_allocation = MemoryAllocation(opt.cuda, opt.batchSize, opt.input_nc, opt.output_nc, opt.size, opt.gpu_ids)
    # test_gd(module, data_gd_loader, data_c_loader, memory_allocation, opt)
    # test_heatmap(module, data_gd_loader, data_c_loader, memory_allocation, opt)