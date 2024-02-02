import argparse
import sys
import os
import torch
import random
import cv2

import torch.nn as nn
import numpy as np
from torchvision.utils import save_image
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from utils import plot_confusion_matrix, plot_roc_auc_curve, save_image

from init import Modules, MemoryAllocation, HingeLoss
from datasets import TestCLoader, TestGDLoader
from utils import FileName
from feature_show import define_model_trunc, plot_2d_features, plot_3d_features
from gradcam import GradCAM, show_cam_on_image
from PIL import Image

import warnings
warnings.filterwarnings("ignore")


def test_cam(module, dataloader, memory_allocation, opt):  # synthetic images from generators
    result_test_path = 'results/{}/{}'.format(opt.dataset_name, opt.project_name) +'/test'
    camA_path = result_test_path + '/GradCamA'
    camB_path = result_test_path + '/GradCamB'
    if not os.path.exists(camA_path):
        os.makedirs(camA_path)
    if not os.path.exists(camB_path):
        os.makedirs(camB_path)

    for i, batch in enumerate(dataloader.test_loader):
        real_A = Variable(memory_allocation.input_A.copy_(batch['X']))    # Set model input realA
        real_B = Variable(memory_allocation.input_B.copy_(batch['Y']))    # Set model input realB

        imgA = Image.open(batch['X_path'][0]).convert('RGB')
        imgA = np.array(imgA, dtype=np.uint8)
        imgA = cv2.resize(imgA, (224, 224))
        grayscale_camA = cam(input_tensor=real_A, target_category=0)
        grayscale_camA = grayscale_camA[0, :]  # [224, 224]
        visualizationA = show_cam_on_image(imgA.astype(dtype=np.float32) / 255., grayscale_camA, use_rgb=True)
        A_name = FileName(batch['X_path'][0])
        save_image(visualizationA, camA_path + '/{}png'.format(A_name))

        imgB = Image.open(batch['Y_path'][0]).convert('RGB')
        imgB = np.array(imgB, dtype=np.uint8)
        imgB = cv2.resize(imgB, (224, 224))
        grayscale_camB = cam(input_tensor=real_B, target_category=1)
        grayscale_camB = grayscale_camB[0, :]  # [224, 224]
        visualizationB = show_cam_on_image(imgB.astype(dtype=np.float32) / 255., grayscale_camB, use_rgb=True)
        B_name = FileName(batch['Y_path'][0])
        save_image(visualizationB, camB_path + '/{}png'.format(B_name))
                    
        sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader.test_loader)))
    sys.stdout.write('\n')


def test_c(module, dataloader, opt):
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

    criterion = nn.CrossEntropyLoss()          # define the cost function

    for data, target in tqdm(dataloader.test_loader):
        data, target = data.cuda(opt.gpu_ids[0]), target.cuda(opt.gpu_ids[0])    # move to GPU or cpu
        output = module.netC(data).squeeze(1)                                    # forward pass
        
        loss = criterion(output, target)                     # calculate the loss
        test_loss += loss.item()*data.size(0)                # update testing loss

        prob = torch.softmax(output, dim=1)[:, 1]        # output probabilities for plotting roc curve
        predict_y = torch.max(output, dim=1)[1]         # output predicted class (i.e., idx)

        test_correct += (predict_y == target).sum().item()   # update validation correct numbers
        
        correct_tensor = (predict_y == target)
        for i in range(opt.batchSize):
            c = target[i]
            class_correct[c] += correct_tensor[i].item()
            class_total[c] += 1

            y_true_ = np.squeeze(target[i].data.cpu().float().numpy())
            y_true.append(int(y_true_))
            y_prob_ = np.squeeze(prob[i].data.cpu().float().numpy())
            y_prob.append(y_prob_)

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
    plot_roc_auc_curve(y_true, y_prob, 'resnet18', result_path)
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
    parser.add_argument('--dataroot', type=str, default='/mnt/evo1/xiangyu/icassp2024/icassp_select_covid', help='root directory of the dataset')
    parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=224, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--dataset_name', default='covid1', type=str, help='Choose the dataset name for save results')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of classes for classification')
    parser.add_argument('--gpu_ids', type=str, default='2')
    parser.add_argument('--classifier', type=str, default='checkpoints/covid1/convnext_tiny/convnext_tiny_best_netC.pth', help='classifier checkpoint file')
    parser.add_argument('--project_name', default='convnext_tiny', type=str, help='Choose the project name for save results')
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
    module.netC.load_state_dict(torch.load(opt.classifier))

    # Set module networks test mode
    module.netC.eval()
    # GradCAM attention
    target_layers = [module.netC.module.features[-1]]
    cam = GradCAM(model=module.netC, target_layers=target_layers, use_cuda=False)
    
    # test the cyclegan (Generator & Disciminator) performance
    memory_allocation = MemoryAllocation(opt.cuda, opt.batchSize, opt.input_nc, opt.output_nc, opt.size, opt.gpu_ids)

    # test the classifier performance
    data_c_loader = TestCLoader(opt.size, opt.dataroot, opt.batchSize, opt.n_cpu)
    data_cam_loader = TestGDLoader(opt.size, opt.dataroot, opt.batchSize, opt.n_cpu)
    
    # test_c(module, data_c_loader, opt)
    test_cam(module, data_cam_loader, memory_allocation, opt)