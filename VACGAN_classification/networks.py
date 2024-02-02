import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


class Generator(nn.Module):
    def __init__(self, img_size, channels, latent_dim, num_classes):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(num_classes, latent_dim)

        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img
    

class Discriminator(nn.Module):
    def __init__(self, img_size, channels, num_classes):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


def define_pretrained_model(model_name, num_classes):
    """
    The following classification models are available, 
    with or without pre-trained weights:
    """
    model = None
        
    #------------------------------------AlexNet (2012)------------------------------------#
    if model_name == 'alexnet':   
        model = models.alexnet(weights = models.AlexNet_Weights.DEFAULT)
        num_fc = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features=num_fc, out_features=num_classes) 

    #--------------------------------------VGG (2014)--------------------------------------#
    if model_name == 'vgg11':     
        model = models.vgg11(weights = models.VGG11_Weights.DEFAULT)
    if model_name == 'vgg13':     
        model = models.vgg13(weights = models.VGG13_Weights.DEFAULT)
        num_fc = model.classifier[6].in_features            
        model.classifier[6] = nn.Linear(num_fc, num_classes) 
    if model_name == 'vgg16':     
        model = models.vgg16(weights = models.VGG16_Weights.DEFAULT)
        num_fc = model.classifier[6].in_features            
        model.classifier[6] = nn.Linear(num_fc, num_classes) 

    if model_name == 'vgg19':
        model = models.vgg19(weights = models.VGG19_Weights.DEFAULT)
        num_fc = model.classifier[6].in_features             
        model.classifier[6] = nn.Linear(num_fc, num_classes) 

    #-----------------------------------GoogleNet (2014)-----------------------------------#
    if model_name == 'googlenet':
        model = models.googlenet(weights = models.GoogLeNet_Weights.DEFAULT)
        num_fits = model.fc.in_features
        model.fc = nn.Linear(num_fits, num_classes)

    #-------------------------------------ResNet (2015)------------------------------------#
    if model_name == 'resnet18':
        model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
        num_fits = model.fc.in_features
        model.fc = nn.Linear(num_fits, num_classes) 

    if model_name == 'resnet34':
        model = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)
        num_fits = model.fc.in_features
        model.fc = nn.Linear(num_fits, num_classes)

    if model_name == 'resnet50':
        model = models.resnet50(weights = models.ResNet50_Weights.DEFAULT)
        num_fits = model.fc.in_features
        model.fc = nn.Linear(num_fits, num_classes) 

    if model_name == 'resnet101':
        model = models.resnet101(weights = models.ResNet101_Weights.DEFAULT)

    #----------------------------------Inception v3 (2015)---------------------------------#
    if model_name == 'inception_v3':
        model = models.inception_v3(weights = models.Inception_V3_Weights.DEFAULT)
        model.aux_logits = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs,num_classes)
    #-----------------------------------SqueezeNet (2016)----------------------------------#
    if model_name == 'squeezenet1_0':
        model = models.squeezenet1_0(weights = models.SqueezeNet1_0_Weights.DEFAULT)
    if model_name == 'squeezenet1_1':
        model = models.squeezenet1_1(weights = models.SqueezeNet1_1_Weights.DEFAULT)

    #------------------------------------DenseNet (2016)-----------------------------------#
    if model_name == 'densenet121':
        model = models.densenet121(weights = models.DenseNet121_Weights.DEFAULT)
        model.features.avgpool = nn.AdaptiveAvgPool2d((1,1))
        model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=num_classes)
    if model_name == 'densenet161':
        model = models.densenet161(weights = models.DenseNet161_Weights.DEFAULT)
    if model_name == 'densenet169':
        model = models.densenet169(weights = models.DenseNet169_Weights.DEFAULT)
    if model_name == 'densenet201':
        model = models.densenet201(weights = models.DenseNet201_Weights.DEFAULT)

    #----------------------------------ShuffleNet v2 (2018)--------------------------------#
    if model_name == 'shufflenet_v2_x0_5':  # x0_5, x1_0, x1_5, x2_0
        model = models.shufflenet_v2_x0_5(weights = models.ShuffleNet_V2_X0_5_Weights.DEFAULT)

    #-------------------------------------MnasNet (2018)--------------------------------#
    if model_name == 'mnasnet0_5':     # 0_5, 0_75, 1_0, 1_3
        model = models.mnasnet0_5(weights = models.MNASNet0_5_Weights.DEFAULT)
    if model_name == 'mnasnet1_0':     # 0_5, 0_75, 1_0, 1_3
        model = models.mnasnet1_0(weights = models.MNASNet1_0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)

    #-------------------------------------ResNeXt (2019)-----------------------------------#
    if model_name == 'resnext50_32x4d':
        model = models.resnext50_32x4d(weights = models.ResNeXt50_32X4D_Weights.DEFAULT)
        model = models.resnext101_32x8d(weights = models.ResNeXt101_32X8D_Weights.DEFAULT)
        model = models.resnext101_64x4d(weights = models.ResNeXt101_64X4D_Weights.DEFAULT)

    #------------------------------------MobileNet (2019)----------------------------------#
    if model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights = models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)
    if model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights = models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(in_features=model.classifier[3].in_features, out_features=num_classes)
        model_corr = create_feature_extractor(model, return_nodes={"classifier.2": "coor"})
    if model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights = models.MobileNet_V3_Large_Weights.DEFAULT)

    #-----------------------------------EfficientNet (2019)--------------------------------#
    if model_name == 'efficientnet_b5':  # b0 ~ b7
        model = models.efficientnet_b5(weights = models.EfficientNet_B5_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=num_classes)

    #--------------------------------------RegNet (2020)-----------------------------------#
    if model_name == 'regnet_x_400mf': # X: 400MF, 800MF, 1.6GF, 3.2GF, 8.0GF, 16GF, 32GF
        model = models.regnet_x_400mf(weights = models.RegNet_X_400MF_Weights.DEFAULT)
    if model_name == 'regnet_y_400mf': # Y: 400MF, 800MF, 1.6GF, 3.2GF, 8.0GF, 16GF, 32GF, 128GF
        model = models.regnet_y_400mf(weights = models.RegNet_Y_400MF_Weights.DEFAULT)

    #--------------------------------Vision Transformer (2020)-----------------------------#
    if model_name == 'vit_b_32':  # b_32, b_16, l_32, l_16, h_14
        model = models.vit_b_32(weights = models.vit_b_32)

    #---------------------------------EfficientNet v2 (2021)-------------------------------#
    if model_name == 'efficientnet_v2_s':  # S, M, L
        model = models.efficientnet_v2_s(weights = models.EfficientNet_V2_S_Weights.DEFAULT)

    #-------------------------------------ConvNeXt (2022)----------------------------------#
    if model_name == 'convnext_tiny':  # tiny, small, base, large
        model = models.convnext_tiny(weights = models.ConvNeXt_Tiny_Weights.DEFAULT)
        num_fc = model.classifier[2].in_features 
        model.classifier[2] = nn.Linear(num_fc, out_features = num_classes)
    
    return model
