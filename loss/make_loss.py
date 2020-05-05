import torch
import torch.nn.functional as F
import torch.nn as nn
from .softmax_loss import CrossEntropyLabelSmooth
from .center_loss import CenterLoss
from .triplet_loss import TripletLoss
from .local_loss import LocalLoss
from .aligned_loss import TripletLossAlignedReID

def make_loss(cfg, num_classes):    # modified by gu
    feat_dim = 2048

    if 'triplet' in cfg.LOSS_TYPE:
        triplet = TripletLoss(cfg.MARGIN, cfg.HARD_FACTOR)  # triplet loss

    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    # if 'softmax' in cfg.LOSS_TYPE:
    #     if cfg.LOSS_LABELSMOOTH == 'on':
    #         xent = CrossEntropyLabelSmooth(num_classes=num_classes)  # new add by luo
    #         print("label smooth on, numclasses:", num_classes)
    xent = CrossEntropyLabelSmooth(num_classes=num_classes)
    if 'pcb' in cfg.LOSS_TYPE:
        if cfg.LOSS_LABELSMOOTH == 'on':
            pcb = CrossEntropyLabelSmooth(num_classes=num_classes)  # new add by luo
            print("label smooth on, numclasses:", num_classes)

    # if 'local' in cfg.LOSS_TYPE:
    # local_loss = LocalLoss(cfg.MARGIN, cfg.HARD_FACTOR)

    if 'aligned' in cfg.LOSS_TYPE:
        aligned_loss = TripletLossAlignedReID(margin=0.3)

    def loss_func(score, feat, local_feat, target, pcb_f=None):
        if cfg.LOSS_TYPE == 'triplet+softmax+center':
            #print('Train with center loss, the loss type is triplet+center_loss')
            if cfg.LOSS_LABELSMOOTH == 'on':
                return cfg.CE_LOSS_WEIGHT * xent(score, target) + \
                       cfg.TRIPLET_LOSS_WEIGHT * triplet(feat, target)[0] + \
                       cfg.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return cfg.CE_LOSS_WEIGHT * F.cross_entropy(score, target) + \
                       cfg.TRIPLET_LOSS_WEIGHT * triplet(feat, target)[0] + \
                       cfg.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
        elif cfg.LOSS_TYPE == 'softmax+center':
            #print('Train with center loss, the loss type is triplet+center_loss')
            if cfg.LOSS_LABELSMOOTH == 'on':
                return cfg.CE_LOSS_WEIGHT * xent(score, target) + \
                       cfg.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
            else:
                return cfg.CE_LOSS_WEIGHT * F.cross_entropy(score, target) + \
                       cfg.CENTER_LOSS_WEIGHT * center_criterion(feat, target)
        elif cfg.LOSS_TYPE == 'triplet+softmax':
            #print('Train with center loss, the loss type is triplet+center_loss')
            if cfg.LOSS_LABELSMOOTH == 'on':
                return cfg.CE_LOSS_WEIGHT * xent(score, target) + \
                       cfg.TRIPLET_LOSS_WEIGHT * triplet(feat, target)[0], \
                       F.cross_entropy(score, target),\
                       triplet(feat, target)[0]
            else:
                return cfg.CE_LOSS_WEIGHT * F.cross_entropy(score, target) + \
                       cfg.TRIPLET_LOSS_WEIGHT * triplet(feat, target)[0], \
                       F.cross_entropy(score, target),\
                       triplet(feat, target)[0]

        elif cfg.LOSS_TYPE == "softmax+triplet+aligned":
            global_loss, local_loss = aligned_loss(feat, target, local_feat)
            if cfg.LOSS_LABELSMOOTH == 'on':
                return cfg.CE_LOSS_WEIGHT * xent(score, target) + \
                       cfg.TRIPLET_LOSS_WEIGHT * global_loss + \
                       cfg.LOCAL_LOSS_WEIGHT * local_loss, \
                       cfg.CE_LOSS_WEIGHT * xent(score, target), \
                       cfg.TRIPLET_LOSS_WEIGHT * global_loss, \
                       cfg.LOCAL_LOSS_WEIGHT * local_loss
                   
        elif cfg.LOSS_TYPE == 'aligned+pcb':
            global_loss, local_loss = aligned_loss(feat, target, local_feat)
            if cfg.LOSS_LABELSMOOTH == 'on':
                sm = nn.Softmax(dim=1)
                loss = 0.
                score = 0
                for x in pcb_f:
                    loss += pcb(x, target)
                    score += sm(x)
                _, preds = torch.max(score.data, 1)
                loss /= len(pcb_f)
            return cfg.CE_LOSS_WEIGHT * loss + \
                   cfg.TRIPLET_LOSS_WEIGHT * global_loss + \
                   cfg.LOCAL_LOSS_WEIGHT * local_loss, \
                   cfg.CE_LOSS_WEIGHT * loss, \
                   cfg.TRIPLET_LOSS_WEIGHT * global_loss, \
                   cfg.LOCAL_LOSS_WEIGHT * local_loss, \
                   preds
        elif cfg.LOSS_TYPE == 'aligned+pcb+center':
            global_loss, local_loss = aligned_loss(feat, target, local_feat)
            if cfg.LOSS_LABELSMOOTH == 'on':
                sm = nn.Softmax(dim=1)
                loss = 0.
                score = 0
                for x in pcb_f:
                    loss += pcb(x, target)
                    score += sm(x)
                _, preds = torch.max(score.data, 1)
                loss /= len(pcb_f)
            return cfg.CE_LOSS_WEIGHT * loss + \
                   cfg.TRIPLET_LOSS_WEIGHT * global_loss + \
                   cfg.LOCAL_LOSS_WEIGHT * local_loss + \
                   cfg.CENTER_LOSS_WEIGHT * center_criterion(feat, target), \
                   cfg.CE_LOSS_WEIGHT * loss, \
                   cfg.TRIPLET_LOSS_WEIGHT * global_loss, \
                   cfg.LOCAL_LOSS_WEIGHT * local_loss, \
                   cfg.CENTER_LOSS_WEIGHT * center_criterion(feat, target), \
                   preds
        elif cfg.LOSS_TYPE == "aligned+arcface":
            global_loss, local_loss = aligned_loss(feat, target, local_feat)
            if cfg.LOSS_LABELSMOOTH == 'on':
                return cfg.CE_LOSS_WEIGHT * xent(score, target) + \
                       cfg.TRIPLET_LOSS_WEIGHT * global_loss + \
                       cfg.LOCAL_LOSS_WEIGHT * local_loss, \
                       cfg.CE_LOSS_WEIGHT * xent(score, target), \
                       cfg.TRIPLET_LOSS_WEIGHT * global_loss, \
                       cfg.LOCAL_LOSS_WEIGHT * local_loss

        elif cfg.LOSS_TYPE == "aligned+arcface+center":
            global_loss, local_loss = aligned_loss(feat, target, local_feat)
            if cfg.LOSS_LABELSMOOTH == 'on':
                return cfg.CE_LOSS_WEIGHT * xent(score, target) + \
                       cfg.TRIPLET_LOSS_WEIGHT * global_loss + \
                       cfg.LOCAL_LOSS_WEIGHT * local_loss + \
                       cfg.CENTER_LOSS_WEIGHT * center_criterion(feat, target), \
                       cfg.CE_LOSS_WEIGHT * xent(score, target), \
                       cfg.TRIPLET_LOSS_WEIGHT * global_loss, \
                       cfg.LOCAL_LOSS_WEIGHT * local_loss, \
                       cfg.CENTER_LOSS_WEIGHT * center_criterion(feat, target)

        elif cfg.LOSS_TYPE == 'softmax':
            if cfg.LOSS_LABELSMOOTH == 'on':
                return xent(score, target)
            else:
                return F.cross_entropy(score, target)
        else:
            print('unexpected loss type')

    return loss_func, center_criterion