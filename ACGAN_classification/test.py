import argparse
import sys
import os
import torch

import numpy as np
from torchvision.utils import save_image
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from utils import plot_confusion_matrix, plot_roc_auc_curve

from init import Modules, MemoryAllocation
from datasets import TestGDLoader, TestCLoader
from utils import FileName
from feature_show import define_model_trunc, plot_2d_features, plot_3d_features

import warnings
warnings.filterwarnings("ignore")


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


    for data, target in tqdm(dataloader.test_loader):
        data, target = data.cuda(opt.gpu_ids[0]), target.cuda(opt.gpu_ids[0])    # move to GPU or cpu
        output = module.netC(data).squeeze(-1)                                  # forward pass

        prob = torch.softmax(output, dim=1)[:, 1]           # output probabilities for plotting roc curve
        predict_y = torch.max(output, dim=1)[1]
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
    parser.add_argument('--dataroot', type=str, default='/mnt/evo1/xiangyu/data/aaai2024/augmented_ultrasound', help='root directory of the dataset')
    parser.add_argument('--channels', type=int, default=3, help='number of channels of input data')
    parser.add_argument('--size', type=int, default=224, help='size of the data (squared assumed)')
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of classes for classification')
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--dataset_name', default='ultrasound3', type=str, help='Choose the dataset name for save results')
    parser.add_argument('--generator', type=str, default='checkpoints/ultrasound3/acgan_convnext_tiny/best_netG.pth', help='generator checkpoint file')
    parser.add_argument('--classifier', type=str, default='checkpoints/ultrasound3/acgan_convnext_tiny/best_netC.pth', help='classifier checkpoint file')
    parser.add_argument('--project_name', default='acgan_convnext_tiny', type=str, help='Choose the project name for save results')
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

    # Load state dicts for generators and a classifier
    module.netG.load_state_dict(torch.load(opt.generator))
    module.netC.load_state_dict(torch.load(opt.classifier))

    # Set module networks test mode
    module.netG.eval()
    module.netC.eval()

    # test the classifier performance
    data_c_loader = TestCLoader(opt.size, opt.dataroot, opt.batchSize, opt.n_cpu)
    test_c(module, data_c_loader, opt)