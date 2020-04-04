import os
import sys
# sys.path.append("/home/vietthang/reid-surveillance-system")
# print(sys.path)
from config import Config9
from datasets import make_dataloader
import torch
from torch.backends import cudnn
import torchvision.transforms as T
from PIL import Image
from utils.logger import setup_logger
from model import make_model

import numpy as np
import cv2
import glob
from utils.metrics import cosine_similarity, euclidean_distance
from utils.distance import low_memory_local_dist

def visualizer(test_img, camid, top_k = 10, img_size=[128,128]):
    figure = np.asarray(query_img.resize((img_size[1],img_size[0])))
    for k in range(top_k):
        name = str(indices[0][k]).zfill(6)
        img = np.asarray(Image.open(img_path[indices[0][k]]).resize((img_size[1],img_size[0])))
        figure = np.hstack((figure, img))
        title=name
    figure = cv2.cvtColor(figure,cv2.COLOR_BGR2RGB)
    if not os.path.exists(cfg.LOG_DIR+ "/results/euclidean"):
        os.mkdir(cfg.LOG_DIR+ "/results/euclidean/")
        print('need to create a new folder named results in {}'.format(cfg.LOG_DIR))
    cv2.imwrite(cfg.LOG_DIR+ "/results/euclidean/{}-cam{}.png".format(os.path.basename(test_img),camid),figure)

if __name__ == "__main__":
    cfg = Config9()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cudnn.benchmark = True

    # model = make_model(Cfg, 255)
    # model.load_param(Cfg.TEST_WEIGHT)
    # train_loader, val_loader, num_query, num_classes = make_dataloader(cfg)
    model = make_model(cfg, 255)
    device = 'cuda'
    model = model.to(device)
    # if device:
    #     if torch.cuda.device_count() > 1:
    #         print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
    #         model = nn.DataParallel(model)
    # model.to(device)
    transform = T.Compose([
        T.Resize(cfg.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    log_dir = cfg.LOG_DIR
    logger = setup_logger('{}.test'.format(cfg.PROJECT_NAME), log_dir)
    model.eval()
    query_images = glob.glob(cfg.QUERY_DIR + '*.jpg')
    print(model)
    print(cfg.QUERY_DIR + '*.png')
    # for test_img in os.listdir(cfg.QUERY_DIR):
    for test_img in query_images:
        logger.info('Finding ID {} ...'.format(test_img))

        gallery_feats = torch.load(cfg.LOG_DIR + '/gfeats.pth')
        gallery_local_feats = torch.load(os.path.join(cfg.LOG_DIR, cfg.GL_FEATS)) 
        img_path = np.load(cfg.LOG_DIR + '/imgpath.npy')
        print(gallery_feats.shape, len(img_path))
        # query_img = Image.open(cfg.QUERY_DIR + test_img)
        query_img = Image.open(test_img)
        input = torch.unsqueeze(transform(query_img), 0)
        input = input.to(device)
        with torch.no_grad():
            x, query_feat, query_local_feat = model(input)
        print(x.shape)
        # dist_mat = cosine_similarity(query_feat, gallery_feats)
        dist_mat = euclidean_distance(query_feat, gallery_feats)
        # print(query_local_feat.shape, gallery_local_feats.shape)
        query_local_feat = query_local_feat.permute(0,2,1)
        local_distmat = low_memory_local_dist(query_local_feat.numpy(),gallery_local_feats.numpy())
        dist_mat = dist_mat + local_distmat
        indices = np.argsort(dist_mat, axis=1)
        visualizer(test_img, camid='mixed', top_k=10, img_size=cfg.INPUT_SIZE)