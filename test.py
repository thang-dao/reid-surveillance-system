import os
from torch.backends import cudnn
from config import  Config_iresnet101_CUHK03_LABELED_CT
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger

if __name__ == "__main__":  
    cfg = Config_iresnet101_CUHK03_LABELED_CT()
    log_dir = cfg.LOG_DIR
    # logger = setup_logger('{}.test'.format(cfg.PROJECT_NAME), log_dir)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes = make_dataloader(cfg)
    model = make_model(cfg, num_classes)
    # print(model)
    if cfg.TEST_MULTIPLE:
        for i in range(100, 200, 5):
            test_weight = cfg.TEST_WEIGHT
            t = test_weight.split('_')
            test_weight = ""
            for k in range(0, len(t)-1):
                test_weight += "{}_".format(t[k])
            test_weight += "{}.pth".format(i)
            print(test_weight)    
            model.load_param(test_weight)

            do_inference(cfg,
                        model,
                        val_loader,
                        num_query, i)
    else:
        model.load_param(cfg.TEST_WEIGHT)
        do_inference(cfg,
                        model,
                        val_loader,
                        num_query)