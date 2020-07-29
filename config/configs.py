from .default import DefaultConfig

class Config_resnet50_Market1501(DefaultConfig):
    """
    Config for aligned algorithm combine PCB: global+local, 
    No last stride then downsample 
    Test:
    Global Distance: mAP:79.2% , Rank1:91.2%  Rank5:96.5%, Rank10: 97.9%, @epoch:200
    Global Distance + Local Distance: mAP:80.6% , Rank1:91.5%  Rank5:97.1%, Rank10: 98.1%, @epoch:200
    """

    def __init__(self):
        super(Config_resnet50_Market1501, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB'
        self.MODEL_NAME = 'resnet50'
        self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/aligned_pcb_Market1501'
        self.OUTPUT_DIR = './output/aligned_pcb_Market1501'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 64
        self.CHECKPOINT_PERIOD = 5
        self.LOSS_TYPE = 'aligned+pcb'
        self.TEST_WEIGHT = './output/aligned_pcb_Market1501/resnet50_50.pth'
        self.LAST_STRIDE = 1
        self.EVAL_PERIOD = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'euclidean'
        self.QUERY_DIR = '/home/vietthang/datasets/Market-1501-v15.09.15/query/'
        self.TEST_MULTIPLE = False
        
class Config_resnet50_Market1501_CT(DefaultConfig):
    """
    Config for aligned algorithm: global+local+center
    No last stride then downsample 
    Test:
    # Global Distance: mAP:79.2% , Rank1:91.2%  Rank5:96.5%, Rank10: 97.9%, @epoch:200
    Global Distance + Local Distance: mAP:81.4% , Rank1:92.5%  Rank5:97.1%, Rank10: 98.1%, @epoch:80
    """

    def __init__(self):
        super(Config_resnet50_Market1501_CT, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB add Center Loss'
        self.MODEL_NAME = 'resnet50'
        self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/aligned_pcb_center_Market1501'
        self.OUTPUT_DIR = './output/aligned_pcb_center_Market1501'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 1
        self.LOSS_TYPE = 'aligned+pcb+center'
        self.TEST_WEIGHT = './output/aligned_pcb_center_Market1501/resnet50_50.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False   
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'euclidean' 

# DukeMTMCREID 
class Config_resnet50_DukeMTMCREID_CT(DefaultConfig):
    """
    Config for aligned algorithm: global+local+center
    No last stride then downsample 
    Test:
    # Global Distance: 
    Global Distance + Local Distance: 
    mAP: 75.3%, Rank1: 85.3%, Rank5: 93.4%, Rank10: 95.5%, @epoch: 80
    """

    def __init__(self):
        super(Config_resnet50_DukeMTMCREID_CT, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB add Center Loss'
        self.MODEL_NAME = 'resnet50'
        # self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.DATA_NAME = 'dukemtmcreid'
        self.LOG_DIR = './log/aligned_pcb_center_DukeMTMCREID'
        self.OUTPUT_DIR = './output/aligned_pcb_center_DukeMTMCREID'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.LOSS_TYPE = 'aligned+pcb+center'
        self.TEST_WEIGHT = './output/aligned_pcb_center_DukeMTMCREID/resnet50_80.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'cosine' 
        self.QUERY_DIR = '/home/vietthang/datasets/DukeMTMC-reID/query/'

class Config_resnet50_DukeMTMCREID(DefaultConfig):
    """
    Config for aligned algorithm: global+local, 
    No last stride then downsample 
    Test:
    Global Distance + Local Distance: mAP: 73.7%, Rank1:85.1%  Rank5:93.0%, Rank10: 94.9%, @epoch:80
    """

    def __init__(self):
        super(Config_resnet50_DukeMTMCREID, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB'
        self.MODEL_NAME = 'resnet50'
        self.DATA_NAME = 'dukemtmcreid'
        # self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/aligned_pcb_DukeMTMCREID'
        self.OUTPUT_DIR = './output/aligned_pcb_DukeMTMCREID'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.LOSS_TYPE = 'aligned+pcb'
        self.TEST_WEIGHT = './output/aligned_pcb_DukeMTMCREID/resnet50_60.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'cosine'
        self.QUERY_DIR = '/home/vietthang/datasets/DukeMTMC-reID/query/'

#CUHK03
class Config_resnet50_CUHK03(DefaultConfig):
    """
    Config for aligned algorithm: global+local, 
    No last stride then downsample 
    Test:
    Global Distance + Local Distance: mAP: 73.7%, Rank1:85.1%  Rank5:93.0%, Rank10: 94.9%, @epoch:80
    """

    def __init__(self):
        super(Config_resnet50_CUHK03, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB'
        self.MODEL_NAME = 'resnet50'
        self.DATA_NAME = 'cuhk03'
        # self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/aligned_pcb_CUHK03'
        self.OUTPUT_DIR = './output/aligned_pcb_CUHK03'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.EVAL_METRIC = 'cuhk03'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.EVAL_PERIOD = 20
        self.LOSS_TYPE = 'aligned+pcb'
        self.TEST_WEIGHT = './output/aligned_pcb_CUHK03/resnet50_90.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'euclidean'
        self.cuhk03_labeled = False

class Config_resnet50_CUHK03_CT(DefaultConfig):
    """
    Config for aligned algorithm: global+local, 
    No last stride then downsample 
    Test:
    Global Distance + Local Distance: 
    """

    def __init__(self):
        super(Config_resnet50_CUHK03_CT, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB and Center'
        self.MODEL_NAME = 'resnet50'
        self.DATA_NAME = 'cuhk03'
        # self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/aligned_pcb_center_CUHK03'
        self.OUTPUT_DIR = './output/aligned_pcb_center_CUHK03'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.EVAL_METRIC = 'cuhk03'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.EVAL_PERIOD = 20
        self.LOSS_TYPE = 'aligned+pcb+center'
        self.TEST_WEIGHT = './output/aligned_pcb_center_CUHK03/resnet50_75.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]
        self.TEST_IMS_PER_BATCH = 128
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'euclidean'
        self.cuhk03_labeled = False


class Config_resnet50_CUHK03(DefaultConfig):
    def __init__(self):
        super(Config_resnet50_CUHK03, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB'
        self.MODEL_NAME = 'resnet50'
        self.DATA_NAME = 'cuhk03'
        # self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/aligned_pcb_CUHK03'
        self.OUTPUT_DIR = './output/aligned_pcb_CUHK03'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.EVAL_METRIC = 'cuhk03'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.EVAL_PERIOD = 20
        self.LOSS_TYPE = 'aligned+pcb'
        self.TEST_WEIGHT = './output/aligned_pcb_CUHK03/resnet50_90.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'euclidean'
        # self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.DATA_NAME = 'dukemtmcreid'
        self.LOG_DIR = './log/aligned_triplet_softmax_DukeMTMCREID'
        self.OUTPUT_DIR = './output/aligned_triplet_softmax_DukeMTMCREID'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.LOSS_TYPE = 'softmax+triplet+aligned'
        self.TEST_WEIGHT = './output/aligned_triplet_softmax_DukeMTMCREID/resnet50_80.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'euclidean' 
        self.TEST_MULTIPLE = True
        self.cuhk03_labeled = False

# Iresnet 101
class Config_iresnet101_Market(DefaultConfig):
    """
    Config for aligned algorithm combine PCB: global+local, 
    No last stride then downsample 
    Test:
    Global Distance: mAP:79.2% , Rank1:91.2%  Rank5:96.5%, Rank10: 97.9%, @epoch:200
    Global Distance + Local Distance: mAP:80.6% , Rank1:91.5%  Rank5:97.1%, Rank10: 98.1%, @epoch:200
    """

    def __init__(self):
        super(Config_iresnet101_Market, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB'
        self.MODEL_NAME = 'iresnet101'
        self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/iresnet_aligned_pcb_Market1501'
        self.OUTPUT_DIR = './output/iresnet_aligned_pcb_Market1501'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/iresnet-19c8e357.pth'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.LOSS_TYPE = 'aligned+pcb'
        self.TEST_WEIGHT = './output/iresnet_aligned_pcb_Market1501/iresnet_105.pth'
        self.LAST_STRIDE = 1
        self.EVAL_PERIOD = 5
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'euclidean'
        self.QUERY_DIR = '/home/vietthang/datasets/Market-1501-v15.09.15/query/'
        self.TEST_MULTIPLE = False
        self.QUERY_DIR = '/media/leo/data/mydata/CUHK03_LABELED/dataset/Market-1501-v15.09.15/query'

class Config_iresnet101_Market1501_CT(DefaultConfig):
    """
    Config for aligned algorithm: global+local+center
    No last stride then downsample 
    Test:
    # Global Distance: mAP:79.2% , Rank1:91.2%  Rank5:96.5%, Rank10: 97.9%, @epoch:200
    Global Distance + Local Distance: mAP:81.4% , Rank1:92.5%  Rank5:97.1%, Rank10: 98.1%, @epoch:80
    """

    def __init__(self):
        super(Config_iresnet101_Market1501_CT, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB add Center Loss'
        self.MODEL_NAME = 'iresnet101'
        self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/iresnet_aligned_pcb_center_Market1501'
        self.OUTPUT_DIR = './output/iresnet_aligned_pcb_center_Market1501'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.LOSS_TYPE = 'aligned+pcb+center'
        self.TEST_WEIGHT = './output/iresnet_aligned_pcb_center_Market1501/iresnet101_75.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = True   
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'euclidean' 
        self.TEST_MULTIPLE = False

class Config_iresnet101_DukeMTMCREID_CT(DefaultConfig):
    """
    Config for aligned algorithm: global+local+center
    No last stride then downsample 
    Test:
    # Global Distance: 
    Global Distance + Local Distance: 
    mAP: 75.3%, Rank1: 85.3%, Rank5: 93.4%, Rank10: 95.5%, @epoch: 80
    """

    def __init__(self):
        super(Config_iresnet101_DukeMTMCREID_CT, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB add Center Loss'
        self.MODEL_NAME = 'iresnet101'
        # self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.DATA_NAME = 'dukemtmcreid'
        self.LOG_DIR = './log/iresnet_aligned_pcb_center_DukeMTMCREID'
        self.OUTPUT_DIR = './output/iresnet_aligned_pcb_center_DukeMTMCREID'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.LOSS_TYPE = 'aligned+pcb+center'
        self.TEST_WEIGHT = './output/iresnet_aligned_pcb_center_DukeMTMCREID/iresnet_90.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'cosine' 
        self.QUERY_DIR = '/home/vietthang/datasets/DukeMTMC-reID/query/'
        self.TEST_MULTIPLE = False

class Config_iresnet101_DukeMTMCREID(DefaultConfig):
    """
    Config for aligned algorithm: global+local, 
    No last stride then downsample 
    Test:
    Global Distance + Local Distance: mAP: 73.7%, Rank1:85.1%  Rank5:93.0%, Rank10: 94.9%, @epoch:80
    """
    def __init__(self):
        super(Config_iresnet101_DukeMTMCREID, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB'
        self.MODEL_NAME = 'iresnet101'
        self.DATA_NAME = 'dukemtmcreid'
        # self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/iresnet_aligned_pcb_DukeMTMCREID'
        self.OUTPUT_DIR = './output/iresnet_aligned_pcb_DukeMTMCREID'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.LOSS_TYPE = 'aligned+pcb'
        self.TEST_WEIGHT = './output/iresnet_aligned_pcb_DukeMTMCREID/iresnet101_75.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'cosine'
        self.QUERY_DIR = '/home/vietthang/datasets/DukeMTMC-reID/query/'

class Config_iresnet101_CUHK03_DETECTED(DefaultConfig):
    """
    Config for aligned algorithm: global+local, 
    No last stride then downsample 
    Test:
    Global Distance + Local Distance: mAP: 73.7%, Rank1:85.1%  Rank5:93.0%, Rank10: 94.9%, @epoch:80
    """

    def __init__(self):
        super(Config_iresnet101_CUHK03_DETECTED, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB'
        self.MODEL_NAME = 'iresnet101'
        self.DATA_NAME = 'cuhk03'
        # self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/iresnet101_aligned_pcb_CUHK03'
        self.OUTPUT_DIR = './output/iresnet101_aligned_pcb_CUHK03'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.EVAL_METRIC = 'cuhk03'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.EVAL_PERIOD = 20
        self.LOSS_TYPE = 'aligned+pcb'
        self.TEST_WEIGHT = './output/iresnet101_aligned_pcb_CUHK03/iresnet101_185.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = True
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local' 
        self.TEST_METHOD = 'euclidean'
        self.cuhk03_labeled = False
        # self.TEST_MULTIPLE = True
        
class Config_iresnet101_CUHK03_DETECTED_CT(DefaultConfig):
    """
    Config for aligned algorithm: global+local, 
    No last stride then downsample 
    Test:
    Global Distance + Local Distance: 
    """

    def __init__(self):
        super(Config_iresnet101_CUHK03_DETECTED_CT, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB and Center'
        self.MODEL_NAME = 'iresnet101'
        self.DATA_NAME = 'cuhk03'
        # self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/iresnet_aligned_pcb_center_CUHK03'
        self.OUTPUT_DIR = './output/iresnet_aligned_pcb_center_CUHK03'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.EVAL_METRIC = 'cuhk03'
        self.BATCH_SIZE = 64
        self.CHECKPOINT_PERIOD = 5
        self.EVAL_PERIOD = 20
        self.LOSS_TYPE = 'aligned+pcb+center'
        self.TEST_WEIGHT = './output/iresnet_aligned_pcb_center_CUHK03/iresnet_120.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = True
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'euclidean'
        self.cuhk03_labeled = False
        # self.TEST_MULTIPLE = True

class Config_iresnet101_CUHK03_LABELED(DefaultConfig):
    """
    Config for aligned algorithm: global+local, 
    No last stride then downsample 
    Test:
    Global Distance + Local Distance: mAP: 73.7%, Rank1:85.1%  Rank5:93.0%, Rank10: 94.9%, @epoch:80
    """

    def __init__(self):
        super(Config_iresnet101_CUHK03_LABELED, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB'
        self.MODEL_NAME = 'iresnet101'
        self.DATA_NAME = 'cuhk03'
        # self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/iresnet101_aligned_pcb_CUHK03_labeled'
        self.OUTPUT_DIR = './output/iresnet101_aligned_pcb_CUHK03_labeled'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.EVAL_METRIC = 'cuhk03'
        self.BATCH_SIZE = 64
        self.CHECKPOINT_PERIOD = 5
        self.EVAL_PERIOD = 20
        self.LOSS_TYPE = 'aligned+pcb'
        self.TEST_WEIGHT = './output/iresnet101_aligned_pcb_CUHK03_labeled/iresnet101_195.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = True
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'euclidean'
        self.TEST_MULTIPLE = False
        self.cuhk03_labeled = True
        
class Config_iresnet101_CUHK03_LABELED_CT(DefaultConfig):
    """
    Config for aligned algorithm: global+local, 
    No last stride then downsample 
    Test:
    Global Distance + Local Distance: 
    """

    def __init__(self):
        super(Config_iresnet101_CUHK03_LABELED_CT, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB and Center'
        self.MODEL_NAME = 'iresnet101'
        self.DATA_NAME = 'cuhk03'
        # self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/iresnet_aligned_pcb_center_CUHK03_labeled'
        self.OUTPUT_DIR = './output/iresnet_aligned_pcb_center_CUHK03_labeled'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.EVAL_METRIC = 'cuhk03'
        self.BATCH_SIZE = 64
        self.CHECKPOINT_PERIOD = 5
        self.EVAL_PERIOD = 20
        self.LOSS_TYPE = 'aligned+pcb+center'
        self.TEST_WEIGHT = './output/iresnet_aligned_pcb_center_CUHK03_labeled/iresnet101_110.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = True
        self.INPUT_SIZE = [256, 128]
        self.TEST_IMS_PER_BATCH = 128
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'euclidean'
        self.TEST_MULTIPLE = False
        self.cuhk03_labeled = True

# Iresnet50
class Config_iresnet50_Market(DefaultConfig):
    """
    Config for aligned algorithm combine PCB: global+local, 
    No last stride then downsample 
    Test:
    Global Distance: mAP:79.2% , Rank1:91.2%  Rank5:96.5%, Rank10: 97.9%, @epoch:200
    Global Distance + Local Distance: mAP:80.6% , Rank1:91.5%  Rank5:97.1%, Rank10: 98.1%, @epoch:200
    """

    def __init__(self):
        super(Config_iresnet50_Market, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB'
        self.MODEL_NAME = 'iresnet50'
        self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/iresnet50_aligned_pcb_Market1501'
        self.OUTPUT_DIR = './output/iresnet50_aligned_pcb_Market1501'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/iresnet-19c8e357.pth'
        self.BATCH_SIZE = 256
        self.CHECKPOINT_PERIOD = 5
        self.LOSS_TYPE = 'aligned+pcb'
        self.TEST_WEIGHT = './output/iresnet50_aligned_pcb_Market1501/iresnet50_50.pth'
        self.LAST_STRIDE = 1
        self.EVAL_PERIOD = 200
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'euclidean'
        self.QUERY_DIR = '/home/vietthang/datasets/Market-1501-v15.09.15/query/'
        self.TEST_MULTIPLE = True

class Config_iresnet50_DuketMTMCREID_CT(DefaultConfig):
    """
    Config for aligned algorithm: global+local+center
    No last stride then downsample 
    Test:
    # Global Distance: 
    Global Distance + Local Distance: 
    mAP: 75.3%, Rank1: 85.3%, Rank5: 93.4%, Rank10: 95.5%, @epoch: 80
    """

    def __init__(self):
        super(Config_iresnet50_DuketMTMCREID_CT, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB add Center Loss'
        self.MODEL_NAME = 'iresnet50'
        # self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.DATA_NAME = 'dukemtmcreid'
        self.LOG_DIR = './log/iresnet50_aligned_pcb_center_DukeMTMCREID'
        self.OUTPUT_DIR = './output/iresnet50_aligned_pcb_center_DukeMTMCREID'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.LOSS_TYPE = 'aligned+pcb+center'
        self.TEST_WEIGHT = './output/iresnet50_aligned_pcb_center_DukeMTMCREID/iresnet50_50.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'cosine' 
        self.QUERY_DIR = '/home/vietthang/datasets/DukeMTMC-reID/query/'
        self.TEST_MULTIPLE = True

class Config_resnet101_Market(DefaultConfig):
    """
    Config for aligned algorithm combine PCB: global+local, 
    No last stride then downsample 
    Test:
    Global Distance: mAP:79.2% , Rank1:91.2%  Rank5:96.5%, Rank10: 97.9%, @epoch:200
    Global Distance + Local Distance: mAP:80.6% , Rank1:91.5%  Rank5:97.1%, Rank10: 98.1%, @epoch:200
    """

    def __init__(self):
        super(Config_resnet101_Market, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB'
        self.MODEL_NAME = 'resnet101'
        self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/resnet101_aligned_pcb_Market1501'
        self.OUTPUT_DIR = './output/resnet101_aligned_pcb_Market1501'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet101-5d3b4d8f.pth'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.LOSS_TYPE = 'aligned+pcb'
        self.TEST_WEIGHT = './output/resnet101_aligned_pcb_Market1501/resnet101_50.pth'
        self.LAST_STRIDE = 1
        self.EVAL_PERIOD = 5
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'euclidean'
        self.QUERY_DIR = '/home/vietthang/datasets/Market-1501-v15.09.15/query/'
        self.TEST_MULTIPLE = True

class Config_resnet101_DukeMTMCREID(DefaultConfig):
    """
    Config for aligned algorithm: global+local, 
    No last stride then downsample 
    Test:
    Global Distance + Local Distance: mAP: 73.7%, Rank1:85.1%  Rank5:93.0%, Rank10: 94.9%, @epoch:80
    """

    def __init__(self):
        super(Config_resnet101_DukeMTMCREID, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB'
        self.MODEL_NAME = 'resnet101'
        self.DATA_NAME = 'dukemtmcreid'
        self.LOG_DIR = './log/resnet101_aligned_pcb_DukeMTMCREID'
        self.OUTPUT_DIR = './output/resnet101_aligned_pcb_DukeMTMCREID'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet101-5d3b4d8f.pth'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 10
        self.LOSS_TYPE = 'aligned+pcb'
        self.TEST_WEIGHT = './output/resnet101_aligned_pcb_DukeMTMCREID/resnet101_50.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'cosine'
        self.QUERY_DIR = '/home/vietthang/datasets/DukeMTMC-reID/query/'
        self.TEST_MULTIPLE = True

class Config_resnet101_CUHK03_CT(DefaultConfig):
    """
    Config for aligned algorithm: global+local, 
    No last stride then downsample 
    Test:
    Global Distance + Local Distance: 
    """

    def __init__(self):
        super(Config_resnet101_CUHK03_CT, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB and Center'
        self.MODEL_NAME = 'resnet101'
        self.DATA_NAME = 'cuhk03'
        # self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/resnet101_aligned_pcb_center_CUHK03'
        self.OUTPUT_DIR = './output/resnet101_aligned_pcb_center_CUHK03'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet101-5d3b4d8f.pth'
        self.EVAL_METRIC = 'cuhk03'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.EVAL_PERIOD = 20
        self.LOSS_TYPE = 'aligned+pcb+center'
        self.TEST_WEIGHT = './output/resnet101_aligned_pcb_center_CUHK03/resnet101_50.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]
        self.TEST_IMS_PER_BATCH = 128
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'euclidean'
        self.TEST_MULTIPLE = True

class Config_resnet101_DuketMTMCREID_CT(DefaultConfig):
    """
    Config for aligned algorithm: global+local+center
    No last stride then downsample 
    Test:
    # Global Distance: 
    Global Distance + Local Distance: 
    mAP: 75.3%, Rank1: 85.3%, Rank5: 93.4%, Rank10: 95.5%, @epoch: 80
    """

    def __init__(self):
        super(Config_resnet101_DuketMTMCREID_CT, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB add Center Loss'
        self.MODEL_NAME = 'iresnet101'
        # self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.DATA_NAME = 'dukemtmcreid'
        self.LOG_DIR = './log/resnet101_aligned_pcb_center_DukeMTMCREID'
        self.OUTPUT_DIR = './output/resnet101_aligned_pcb_center_DukeMTMCREID'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.LOSS_TYPE = 'aligned+pcb+center'
        self.TEST_WEIGHT = './output/resnet101_aligned_pcb_center_DukeMTMCREID/resnet101_50.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'cosine' 
        self.QUERY_DIR = '/home/vietthang/datasets/DukeMTMC-reID/query/'
        self.TEST_MULTIPLE = False

class Config_iresnet101_DukeMTMCREID(DefaultConfig):
    """
    Config for aligned algorithm: global+local, 
    No last stride then downsample 
    Test:
    Global Distance + Local Distance: mAP: 73.7%, Rank1:85.1%  Rank5:93.0%, Rank10: 94.9%, @epoch:80
    """

    def __init__(self):
        super(Config_iresnet101_DukeMTMCREID, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB'
        self.MODEL_NAME = 'iresnet101'
        self.DATA_NAME = 'dukemtmcreid'
        # self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/iresnet_aligned_pcb_DukeMTMCREID'
        self.OUTPUT_DIR = './output/iresnet_aligned_pcb_DukeMTMCREID'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.LOSS_TYPE = 'aligned+pcb'
        self.TEST_WEIGHT = './output/iresnet_aligned_pcb_DukeMTMCREID/iresnet101_75.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = True
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'cosine'
        self.QUERY_DIR = '/home/vietthang/datasets/DukeMTMC-reID/query/'
        self.TEST_MULTIPLE = False

class Config_iresnet50_CUHK03_CT(DefaultConfig):
    """
    Config for aligned algorithm: global+local, 
    No last stride then downsample 
    Test:
    Global Distance + Local Distance: 
    """

    def __init__(self):
        super(Config_iresnet50_CUHK03_CT, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB and Center'
        self.MODEL_NAME = 'iresnet50'
        self.DATA_NAME = 'cuhk03'
        # self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/iresnet50_aligned_pcb_center_CUHK03'
        self.OUTPUT_DIR = './output/iresnet50_aligned_pcb_center_CUHK03'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet101-5d3b4d8f.pth'
        self.EVAL_METRIC = 'cuhk03'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.EVAL_PERIOD = 20
        self.LOSS_TYPE = 'aligned+pcb+center'
        self.TEST_WEIGHT = './output/iresnet50_aligned_pcb_center_CUHK03/iresnet50_50.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]
        self.TEST_IMS_PER_BATCH = 128
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'euclidean'
        self.TEST_MULTIPLE = True