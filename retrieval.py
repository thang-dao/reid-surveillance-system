import os
import sys
import argparse
import glob
import torch
from torch.backends import cudnn
import torchvision.transforms as T
from torch.nn import functional as F
from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import shutil
from utils.logger import setup_logger
from utils.metrics import cosine_similarity, euclidean_distance
from utils.distance import low_memory_local_dist
from model import make_model
from datasets import make_dataloader
from config import  Config_iresnet101_Market, \
                    Config_iresnet101_Market1501_CT, \
                    Config_iresnet101_DukeMTMCREID, \
                    Config_iresnet101_DukeMTMCREID_CT, \
                    Config_iresnet101_CUHK03_DETECTED, \
                    Config_iresnet101_CUHK03_DETECTED_CT, \
                    Config_iresnet101_CUHK03_LABELED, \
                    Config_iresnet101_CUHK03_LABELED_CT

mapping = {'Config_iresnet101_Market': Config_iresnet101_Market(),
           'Config_iresnet101_Market1501_CT': Config_iresnet101_Market1501_CT(),
           'Config_iresnet101_DukeMTMCREID': Config_iresnet101_DukeMTMCREID(),
           'Config_iresnet101_DukeMTMCREID_CT': Config_iresnet101_DukeMTMCREID(),
           'Config_iresnet101_CUHK03_DETECTED': Config_iresnet101_CUHK03_DETECTED(),
           'Config_iresnet101_CUHK03_DETECTED_CT': Config_iresnet101_CUHK03_DETECTED_CT(),
           'Config_iresnet101_CUHK03_LABELED:': Config_iresnet101_CUHK03_LABELED(),
           'Config_iresnet101_CUHK03_LABELED_CT': Config_iresnet101_CUHK03_LABELED_CT()}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRID_SPACING = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pathlog', type=str)
    opt = parser.parse_args()
    # Initial config 
    Cfg = Config_iresnet101_Market()
    log_dir = Cfg.LOG_DIR
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cudnn.benchmark = True
    # Initial model
    model = make_model(Cfg, 2)
    # Load weights
    model.load_param(Cfg.TEST_WEIGHT)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)
    transform = T.Compose([
        T.Resize(Cfg.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Load embedding of gallary set
    opt.pathlog = 'log/hcmus'
    gallery_feats = torch.load(opt.pathlog + '/gfeats.pth')
    gallery_feats = gallery_feats.to(device)
    gallery_local_feats = torch.load(opt.pathlog + '/lfeats.pth')
    camids = np.load(opt.pathlog + '/camids.npy', allow_pickle=True)
    img_path = np.load(opt.pathlog + '/imgpath.npy')
    use_local_distance = False

    for idx, path in enumerate(img_path):
        query_img = Image.open(path)
        input = torch.unsqueeze(transform(query_img), 0)
        input = input.to(device)
        with torch.no_grad():
            x, query_feat, query_local_feat = model(input)
        query_feat.to(device)
        query_local_feat.to(device) 
        dist_mat = cosine_similarity(query_feat, gallery_feats)
        query_local_feat = query_local_feat.permute(0,2,1)
        local_distmat = low_memory_local_dist(query_local_feat.cpu().numpy(), gallery_local_feats.cpu().numpy())
        dist_mat = dist_mat + local_distmat 
        indices = np.argsort(dist_mat, axis=1) 
        pathsave = os.path.join(opt.pathlog + '/result', os.path.basename(path))
        top_result = [img_path[indices[0][k]] for k in range(0,50) if camids[idx] != camids[indices[0][k]]]
        if len(top_result) > 0 :
            print(img_path[idx])
            if not os.path.exists(pathsave):
                os.mkdir(pathsave)
            shutil.copy(img_path[idx], pathsave)
            res = [cv2.rectangle(cv2.resize(cv2.imread(img), (50,100)), (0,0), (50,100), (255,0,0)) for i,img in enumerate(top_result)]
            figure = res[0]
            print(len(res))
            for idx in range(1,min(len(res),3)):
                figure = cv2.hconcat([figure, res[idx]])
            cv2.imwrite(pathsave + '/result.png', figure)
