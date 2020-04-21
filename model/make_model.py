import sys
sys.path.append('./')
import torch
import torch.nn as nn
import torchvision
import numpy as np
from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.iresnet import iresnet101, iresnet50
from loss.arcface import ArcFace

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, model='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, model='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_kaiming_pcb(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier_pcb(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)

class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate, relu=False, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming_pcb)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier_pcb)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x

class DimReduceLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels, nonlinear):
        super(DimReduceLayer, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        
        if nonlinear == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif nonlinear == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        self.last_stride = cfg.LAST_STRIDE
        self.model_path = cfg.PRETRAIN_PATH
        self.cos_layer = cfg.COS_LAYER
        self.num_classes = num_classes
        self.model_name = cfg.MODEL_NAME
        self.pretrain_choice = cfg.PRETRAIN_CHOICE
        self.loss_type = cfg.LOSS_TYPE
    # PCB 
        self.parts = 6
        reduced_dim = 256
        nonlinear = 'relu'
        self.parts_avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.conv5 = DimReduceLayer(512 * Bottleneck.expansion, reduced_dim, nonlinear=nonlinear)
        self.feature_dim = reduced_dim
        self.classifiers = nn.ModuleList([nn.Linear(self.feature_dim, num_classes) for _ in range(self.parts)])

        self._init_params()
        if self.model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=self.last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif self.model_name == 'iresnet101':
            self.in_planes = 2048
            self.base = iresnet101(pretrained=True)
        elif self.model_name == 'iresnet50':
            self.in_planes = 2048
            self.base = iresnet50(pretrained=True)
        else:
            self.in_planes = 2048
            self.base = torchvision.models.resnet50(pretrained=True)
            self.base = nn.Sequential(*list(self.base.children())[:-2])
            print('unsupported backbone! only support resnet50, but got {}'.format(model_name))
        # print(self.base)
        if self.pretrain_choice == 'imagenet' and not 'iresnet' in self.model_name:
            self.base.load_param(self.model_path)
            print('Loading pretrained ImageNet model......')
        self.local_conv_out_channels = 128
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        if self.cos_layer:
            print('using cosine layer')
            self.arcface = ArcFace(self.in_planes, self.num_classes, s=30.0, m=0.50)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)


        self.local_conv = nn.Conv2d(self.in_planes, self.local_conv_out_channels, 1)
        self.local_bn = nn.BatchNorm2d(self.local_conv_out_channels)
        self.local_relu = nn.ReLU(inplace=True)
        # self.horizon_pool = nn.functional.max_pool2d(input=x,kernel_size= (1, inp_size[3]))
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck1d = nn.BatchNorm1d(self.in_planes)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, label=None):
      # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        d_feat = torch.zeros([x.shape[0], x.shape[1], 8, 4])
        if self.last_stride == 1:
            for i in range(8):
                for j in range(4):
                    # d_feat[:,:,i,j] = x[:,:,2*i,2*j].clone()
                    d_feat[:,:,i,j] = x[:,:,2*i,2*j]
                    d_feat = d_feat.cuda()
        else:
            d_feat = x
        if self.loss_type == "softmax+triplet+aligned":
            global_feat = nn.functional.avg_pool2d(x, x.size()[2:])
            global_feat = global_feat.view(global_feat.size(0), -1)
            if self.training:
                local_feat = nn.functional.max_pool2d(input=d_feat,kernel_size= (1, d_feat.size()[3])).cuda()
                local_feat = self.local_relu(self.local_bn(self.local_conv(local_feat)))
                y = self.classifier(global_feat)
                return y, global_feat, local_feat
            else:
                local_feat = nn.functional.max_pool2d(input=d_feat,kernel_size= (1, d_feat.size()[3]))
                local_feat = local_feat.view(local_feat.size()[0:3])
                local_feat = local_feat / torch.pow(local_feat,2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
                return x, global_feat, local_feat

        elif 'aligned+pcb' in self.loss_type:
            global_feat = nn.functional.avg_pool2d(x, x.size()[2:])
            global_feat = global_feat.view(global_feat.size(0), -1)
            v_g = self.parts_avgpool(x) 
            v_g = self.dropout(v_g)
            v_h = self.conv5(v_g)
            y = []
            for i in range(self.parts):
                v_h_i = v_h[:, :, i, :]
                v_h_i = v_h_i.view(v_h_i.size(0), -1)
                y_i = self.classifiers[i](v_h_i)
                y.append(y_i)
            if self.training:
                local_feat = nn.functional.max_pool2d(input=d_feat,kernel_size= (1, d_feat.size()[3])).cuda()
                local_feat = self.local_relu(self.local_bn(self.local_conv(local_feat)))
                preds = self.classifier(global_feat)
                # print(global_feat.requires_grad, local_feat.requires_grad)
                return preds, global_feat, local_feat, y
            else:
                # local_feat = nn.functional.max_pool2d(input=d_feat,kernel_size= (1, d_feat.size()[3])).cuda()
                local_feat = nn.functional.max_pool2d(input=d_feat,kernel_size= (1, d_feat.size()[3]))
                local_feat = local_feat.view(local_feat.size()[0:3])
                local_feat = local_feat / torch.pow(local_feat,2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
                # print(global_feat.requires_grad, local_feat.requires_grad)
                return x, global_feat, local_feat
                # return global_feat, local_feat

        elif 'aligned+arcface' in self.loss_type:
            global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
            global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)
            feat = self.bottleneck(global_feat)
            if self.training:
                local_feat = nn.functional.max_pool2d(input=d_feat,kernel_size= (1, d_feat.size()[3])).cuda()
                local_feat = self.local_relu(self.local_bn(self.local_conv(local_feat)))
                if self.cos_layer:
                    cls_score = self.arcface(feat, label)
                    return cls_score, feat, local_feat
            else:
                local_feat = nn.functional.max_pool2d(input=d_feat,kernel_size= (1, d_feat.size()[3]))
                local_feat = local_feat.view(local_feat.size()[0:3])
                local_feat = local_feat / torch.pow(local_feat,2).sum(dim=1, keepdim=True).clamp(min=1e-12).sqrt()
                return x, feat, local_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            # print(i.find('.'))
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i[7:]].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_iresnet(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            print(i.find('.'))
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

def make_model(cfg, num_class):
    model = Backbone(num_class, cfg)
    return model

if __name__ == "__main__":
    from config import Config9
    config = Config9()
    model = make_model(cfg=config, num_class=715)
    inputs = torch.randn((128, 3, 256, 128))
    print(model(inputs)[0].shape) 
