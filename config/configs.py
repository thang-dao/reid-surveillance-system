from .default import DefaultConfig


# class Config(DefaultConfig):
#     """
#     mAP 85.8, Rank1 94.1, @epoch 175
#     """
#     def __init__(self):
#         super(Config, self).__init__()
#         self.CFG_NAME = 'baseline'
#         self.DATA_DIR = '/nfs/public/datasets/person_reid/Market-1501-v15.09.15'
#         self.PRETRAIN_CHOICE = 'imagenet'
#         self.PRETRAIN_PATH = '/nfs/public/pretrained_models/resnet50-19c8e357.pth'
#
#         self.LOSS_TYPE = 'triplet+softmax+center'
#         self.TEST_WEIGHT = './output/resnet50_175.pth'
#         self.FLIP_FEATS = 'on'


class Config(DefaultConfig):
    """
    Config use softmaxt, triplet, local loss, center and no Harder exemple mining in Local loss
    Local loss: Separate image into 16 part after resnet50 model use last stride
    mAP , Rank1 , @epoch 85
    """

    def __init__(self):
        super(Config, self).__init__()
        self.CFG_NAME = 'softmax_triplet_local_center_256_128'
        self.DATA_DIR = '/home/vietthang/reid/datasets/Market-1501-v15.09.15'
        self.LOG_DIR = './log/softmax_triplet_local_center'
        self.OUTPUT_DIR = './output/softmax_triplet_local_center'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 128

        self.LOSS_TYPE = 'softmax+triplet+local+center'
        self.TEST_WEIGHT = './output/resnet50_185.pth'
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]


# class Config(DefaultConfig):
#     def __init__(self):
#         super(Config, self).__init__()
#         self.CFG_NAME = 'baseline'
#         self.DATA_DIR = '/nfs/public/datasets/person_reid/Market-1501-v15.09.15'
#         self.PRETRAIN_CHOICE = 'imagenet'
#         self.PRETRAIN_PATH = '/nfs/public/pretrained_models/resnet50-19c8e357.pth'
#         self.COS_LAYER = True
#         self.LOSS_TYPE = 'softmax'
#         self.TEST_WEIGHT = './output/resnet50_185.pth'
#         self.FLIP_FEATS = 'off'
#         self.RERANKING = True
