import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from torchvision import datasets
from model import make_model
from config import Config_iresnet101_Market
import argparse

class Kite(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = glob.glob(self.root_dir + '/*.png')
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img = Image.open(self.images[idx])
        return self.transform(img), self.images[idx], os.path.basename(self.images[idx])[1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--pathsave', type=str)
    opt = parser.parse_args()
    cfg = Config_iresnet101_Market()
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    testset = Kite(root_dir=opt.data, transform=val_transforms)
    test_loader = DataLoader(testset, batch_size=64, shuffle=True, num_workers=4)

    model = make_model(cfg, 2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device:
            if torch.cuda.device_count() > 1:
                print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
                model = nn.DataParallel(model)
            model.to(device)
    model.eval()
    model.load_param(cfg.TEST_WEIGHT)

    feats = []
    lf_feats = []
    img_path_list = []
    camids = []
    # print(test_loader.__len__())
    for img, img_path, cam in test_loader:
        with torch.no_grad():
            im = img.to(device)
            _, feat, lf = model(im)
            print(feat.shape)
            feats.append(feat.cpu())
            lf_feats.append(lf.cpu())
            img_path_list.extend(img_path)
            print(cam)
            camids.extend(cam)

    feats = torch.cat(feats, dim=0)
    lf_feats = torch.cat(lf_feats, dim=0)
    lf_feats = lf_feats.permute(0,2,1)
    np.save(os.path.join(opt.pathsave, 'imgpath.npy'), img_path_list)
    np.save(os.path.join(opt.pathsave, 'camids.npy'), camids)
    torch.save(feats, os.path.join(opt.pathsave, 'gfeats.pth'))
    torch.save(lf_feats, os.path.join(opt.pathsave, 'lfeats.pth'))
