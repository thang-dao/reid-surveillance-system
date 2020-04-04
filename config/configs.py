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

class Config9(DefaultConfig):
    """
    Config for aligned algorithm combine PCB: global+local, 
    No last stride then downsample 
    Test:
    Global Distance: mAP:79.2% , Rank1:91.2%  Rank5:96.5%, Rank10: 97.9%, @epoch:200
    Global Distance + Local Distance: mAP:80.6% , Rank1:91.5%  Rank5:97.1%, Rank10: 98.1%, @epoch:200
    """

    def __init__(self):
        super(Config9, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB'
        self.MODEL_NAME = 'resnet50'
        self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/aligned_pcb'
        self.OUTPUT_DIR = './output/aligned_pcb'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 64
        self.CHECKPOINT_PERIOD = 5
        self.LOSS_TYPE = 'aligned+pcb'
        self.TEST_WEIGHT = './output/aligned_pcb/resnet50_50.pth'
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
        
class Config10(DefaultConfig):
    """
    Config for aligned algorithm: global+local+center
    No last stride then downsample 
    Test:
    # Global Distance: mAP:79.2% , Rank1:91.2%  Rank5:96.5%, Rank10: 97.9%, @epoch:200
    Global Distance + Local Distance: mAP:81.4% , Rank1:92.5%  Rank5:97.1%, Rank10: 98.1%, @epoch:80
    """

    def __init__(self):
        super(Config10, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB add Center Loss'
        self.MODEL_NAME = 'resnet50'
        self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/aligned_pcb_center'
        self.OUTPUT_DIR = './output/aligned_pcb_center'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 1
        self.LOSS_TYPE = 'aligned+pcb+center'
        self.TEST_WEIGHT = './output/aligned_pcb_center/resnet50_50.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False   
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'euclidean' 

# DukeMTMCREID 
class Config11(DefaultConfig):
    """
    Config for aligned algorithm: global+local+center
    No last stride then downsample 
    Test:
    # Global Distance: 
    Global Distance + Local Distance: 
    mAP: 75.3%, Rank1: 85.3%, Rank5: 93.4%, Rank10: 95.5%, @epoch: 80
    """

    def __init__(self):
        super(Config11, self).__init__()
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
        self.TEST_METHOD = 'euclidean' 

class Config12(DefaultConfig):
    """
    Config for aligned algorithm: global+local, 
    No last stride then downsample 
    Test:
    Global Distance + Local Distance: mAP: 73.7%, Rank1:85.1%  Rank5:93.0%, Rank10: 94.9%, @epoch:80
    """

    def __init__(self):
        super(Config12, self).__init__()
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
        self.TEST_METHOD = 'euclidean'

#CUHK03
class Config13(DefaultConfig):
    """
    Config for aligned algorithm: global+local, 
    No last stride then downsample 
    Test:
    Global Distance + Local Distance: mAP: 73.7%, Rank1:85.1%  Rank5:93.0%, Rank10: 94.9%, @epoch:80
    """

    def __init__(self):
        super(Config13, self).__init__()
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

class Config14(DefaultConfig):
    """
    Config for aligned algorithm: global+local, 
    No last stride then downsample 
    Test:
    Global Distance + Local Distance: 
    """

    def __init__(self):
        super(Config14, self).__init__()
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

#Experiment with difference PxK
class Config_9_1(DefaultConfig):
    """
    Config for aligned algorithm combine PCB: global+local, 
    No last stride then downsample 
    Test:
    Global Distance: mAP:79.2% , Rank1:91.2%  Rank5:96.5%, Rank10: 97.9%, @epoch:200
    Global Distance + Local Distance: mAP:80.6% , Rank1:91.5%  Rank5:97.1%, Rank10: 98.1%, @epoch:200
    """

    def __init__(self):
        super(Config_9_1, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB with adapt optimizer'
        self.MODEL_NAME = 'resnet50'
        self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/aligned_pcb_PxK_8x3'
        self.OUTPUT_DIR = './output/aligned_pcb_PxK_8x3'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 64
        self.NUM_IMG_PER_ID = 6
        self.MAX_EPOCHS = 100
        self.CHECKPOINT_PERIOD = 5
        self.LOSS_TYPE = 'aligned+pcb'
        self.TEST_WEIGHT = './output/aligned_pcb_PxK_8x3/resnet50_175.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = True
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'euclidean'

class Config15(DefaultConfig):
    """
    Config for aligned algorithm: global+local, 
    No last stride then downsample 
    Test:
    Global Distance + Local Distance: 
    """

    def __init__(self):
        super(Config15, self).__init__()
        self.CFG_NAME = 'Aligned algorithm with Arcface'
        self.MODEL_NAME = 'resnet50'
        self.DATA_NAME = 'market1501'
        # self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/aligned_arface_Market1501'
        self.OUTPUT_DIR = './output/aligned_arface_Market1501'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.EVAL_METRIC = 'market'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.EVAL_PERIOD = 20
        self.LOSS_TYPE = 'aligned+arcface'
        self.COS_LAYER = True
        self.TEST_WEIGHT = './output/aligned_arface_Market1501/resnet50_75.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]
        self.TEST_IMS_PER_BATCH = 128
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'euclidean'
        self.TEST_MULTIPLE = True

class Config16(DefaultConfig):
    """
    Config for aligned algorithm: global+local, 
    No last stride then downsample 
    Test:
    Global Distance + Local Distance: 
    """

    def __init__(self):
        super(Config16, self).__init__()
        self.CFG_NAME = 'Aligned algorithm with Arcface, Center'
        self.MODEL_NAME = 'resnet50'
        self.DATA_NAME = 'market1501'
        # self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/aligned_arface_center_Market1501'
        self.OUTPUT_DIR = './output/aligned_arface_center_Market1501'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.EVAL_METRIC = 'market'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.EVAL_PERIOD = 20
        self.LOSS_TYPE = 'aligned+arcface+center'
        self.COS_LAYER = True
        self.TEST_MULTIPLE = True
        self.TEST_WEIGHT = './output/aligned_arface_center_Market1501/resnet50_75.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]
        self.TEST_IMS_PER_BATCH = 128
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'euclidean'

