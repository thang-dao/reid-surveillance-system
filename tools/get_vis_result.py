import os
import sys
sys.path.append("/home/vietthang/reid-surveillance-system")
# print(sys.path)
from config import Config16
from datasets import make_dataloader
import torch
from torch.backends import cudnn
import torchvision.transforms as T
from PIL import Image
from utils.logger import setup_logger
from model import make_model
from torch.nn import functional as F
import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
from utils.metrics import cosine_similarity, euclidean_distance
from utils.distance import low_memory_local_dist

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRID_SPACING = 10

def visactmap(imgs, outputs):
    height = 256
    width = 128
    img_mean = IMAGENET_MEAN
    img_std = IMAGENET_STD
    # compute activation maps
    outputs = (outputs**2).sum(1)
    b, h, w = outputs.size()
    outputs = outputs.view(b, h * w)
    outputs = F.normalize(outputs, p=2, dim=1)
    outputs = outputs.view(b, h, w)
    imgs, outputs = imgs.cpu(), outputs.cpu()
    img = imgs[0,:,:,:]
    for t, m, s in zip(img, img_mean, img_std):
        t.mul_(s).add_(m).clamp_(0, 1)
    img_np = np.uint8(np.floor(img.numpy() * 255))
    # import pdb; pdb.set_trace()
    img_np = img_np.transpose((1, 2, 0)) # (c, h, w) -> (h, w, c)

    # activation map
    am = outputs[0, ...].detach().numpy()
    am = cv2.resize(am, (width, height))
    am = 255 * (am - np.min(am)) / (
        np.max(am) - np.min(am) + 1e-12
    )
    am = np.uint8(np.floor(am))
    am = cv2.applyColorMap(am, cv2.COLORMAP_JET)

    # overlapped
    overlapped = img_np*0.3 + am*0.7
    overlapped[overlapped > 255] = 255
    overlapped = overlapped.astype(np.uint8)

    # save images in a single figure (add white spacing between images)
    # from left to right: original image, activation map, overlapped image
    grid_img = 255 * np.ones(
        (height, 3*width + 2*GRID_SPACING, 3), dtype=np.uint8
    )
    grid_img[:, :width, :] = img_np[:, :, ::-1]
    grid_img[:,
                width + GRID_SPACING:2*width + GRID_SPACING, :] = am
    grid_img[:, 2*width + 2*GRID_SPACING:, :] = overlapped
    return grid_img

def visualizer(test_img, camid, top_k = 10, img_size=[128,128]):
    figure = np.asarray(query_img.resize((img_size[1],img_size[0])))
    x, _, _ = model(input)
    query_cam = visactmap(input, x)
    figure = np.hstack((figure, query_cam))
    for k in range(top_k):
        name = str(indices[0][k]).zfill(6)
        img = np.asarray(Image.open(img_path[indices[0][k]]).resize((img_size[1],img_size[0])))
        inputs = torch.unsqueeze(transform(Image.open(img_path[indices[0][k]])), 0)
        inputs = inputs.to(device)
        with torch.no_grad():
            x, feat, local_feat = model(inputs)
        img_cam = visactmap(inputs, x)
        figure = np.hstack((figure, img_cam))
        title=name
    figure = cv2.cvtColor(figure,cv2.COLOR_BGR2RGB)
    if not os.path.exists(Cfg.LOG_DIR+ "/results/euclidean"):
        os.mkdir(Cfg.LOG_DIR+ "/results/euclidean/")
        print('need to create a new folder named results in {}'.format(Cfg.LOG_DIR))
    cv2.imwrite(Cfg.LOG_DIR+ "/results/euclidean/{}-cam{}.png".format(os.path.basename(test_img),camid),figure)

if __name__ == "__main__":
    Cfg = Config16()
    log_dir = Cfg.LOG_DIR
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    cudnn.benchmark = True
    model = make_model(Cfg, 255)
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

    log_dir = Cfg.LOG_DIR
    logger = setup_logger('{}.test'.format(Cfg.PROJECT_NAME), log_dir)
    query_images = glob.glob(Cfg.QUERY_DIR + '*.jpg')
    for test_img in query_images:
        logger.info('Finding ID {} ...'.format(test_img))
        gallery_feats = torch.load(Cfg.LOG_DIR + '/gfeats.pth')
        gallery_local_feats = torch.load(os.path.join(Cfg.LOG_DIR, Cfg.GL_FEATS)) 
        img_path = np.load(Cfg.LOG_DIR + '/imgpath.npy')
        print(gallery_feats.shape, len(img_path))
        query_img = Image.open(test_img)
        input = torch.unsqueeze(transform(query_img), 0)
        input = input.to(device)
        with torch.no_grad():
            x, query_feat, query_local_feat = model(input)

        query_feat.to(device)
        query_local_feat.to(device) 
        gallery_feats = gallery_feats.to(device)
        dist_mat = cosine_similarity(query_feat, gallery_feats)
        # dist_mat = euclidean_distance(query_feat, gallery_feats)
        query_local_feat = query_local_feat.permute(0,2,1)
        local_distmat = low_memory_local_dist(query_local_feat.cpu().numpy(),gallery_local_feats.cpu().numpy())
        dist_mat = dist_mat + local_distmat
        indices = np.argsort(dist_mat, axis=1)
        visualizer(test_img, camid='mixed', top_k=10, img_size=Cfg.INPUT_SIZE)