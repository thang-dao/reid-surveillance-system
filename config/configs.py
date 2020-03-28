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
    Config use softmaxt, triplet, local loss, no Harder exemple mining in Local loss
    Local loss: Separate image into 8 part after resnet50 model use last stride
    mAP , Rank1 , @epoch 
    """

    def __init__(self):
        super(Config, self).__init__()
        self.CFG_NAME = 'softmax_triplet_local_no_HEM_256_128'
        self.MODEL_NAME = 'originalResnet50'
        self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/softmax_triplet_local_no_HEM'
        self.OUTPUT_DIR = './output/softmax_triplet_local_no_HEM'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 128

        self.LOSS_TYPE = 'softmax+triplet+local'
        self.TEST_WEIGHT = './output/resnet50_185.pth'
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]

class Config1(DefaultConfig):
    """
    Config use softmaxt, triplet, local loss, and Harder exemple mining in Local loss
    Local loss: Separate image into 8 part after resnet50 model use last stride
    mAP: 83.0% , Rank1: 92.6%, Rank , @epoch 
    """

    def __init__(self):
        super(Config1, self).__init__()
        self.CFG_NAME = 'softmax_triplet_local_256_128'
        self.MODEL_NAME = 'originalResnet50'
        self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/softmax_triplet_local'
        self.OUTPUT_DIR = './output/softmax_triplet_local'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 128

        self.LOSS_TYPE = 'softmax+triplet+local'
        self.TEST_WEIGHT = './output/resnet50_185.pth'
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]

class Config2(DefaultConfig):
    """
    Config use softmaxt, triplet, local loss
    No Harder exemple mining in Local loss
    Local loss Separate image into 8 part after resnet50 model use: last stride, then downsample for local features
    mAP: 85.5% , Rank1: 93.8% , Rank5: 97.9%, Rank10: 98.9%, @epoch: 200 
    """

    def __init__(self):
        super(Config2, self).__init__()
        self.CFG_NAME = 'softmax_triplet_local_HEM_256_128_downsample'
        self.MODEL_NAME = 'resnet50'
        self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/softmax_triplet_local_HEM_downsample'
        self.OUTPUT_DIR = './output/softmax_triplet_local_HEM_downsample'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 128

        self.LOSS_TYPE = 'softmax+triplet+local'
        self.TEST_WEIGHT = './output/resnet50_185.pth'
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]

class Config3(DefaultConfig):
    """
    Config use softmaxt, triplet Loss
    mAP , Rank1 , @epoch 
    """

    def __init__(self):
        super(Config3, self).__init__()
        self.CFG_NAME = 'softmax_triplet_256_128'
        self.MODEL_NAME = 'resnet50'
        self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/softmax_triplet'
        self.OUTPUT_DIR = './output/softmax_triplet'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.LOSS_TYPE = 'triplet+softmax'
        self.TEST_WEIGHT = './output/resnet50_185.pth'
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]

class Config4(DefaultConfig):
    """
    Config use pcb, triplet, local loss
    Harder exemple mining in Local loss
    Local loss Separate image into 8 part after resnet50 model use: last stride, then downsample for local features
    mAP , Rank1 , @epoch 
    """

    def __init__(self):
        super(Config4, self).__init__()
        self.CFG_NAME = 'pcb_triplet_local_HEM_256_128'
        self.MODEL_NAME = 'resnet50'
        self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/pcb_triplet_local_HEM_256_128'
        self.OUTPUT_DIR = './output/pcb_triplet_local_HEM_256_128'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.LOSS_TYPE = 'pcb+triplet+local'
        self.TEST_WEIGHT = './output/resnet50_185.pth'
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]

class Config5(DefaultConfig):
    """
    Config use pcb
    mAP: 74.8% , Rank1: 88.9%, Rank5: 95.7%, Rank10: 97.3% @epoch: 200
    """

    def __init__(self):
        super(Config5, self).__init__()
        self.CFG_NAME = 'pcb'
        self.MODEL_NAME = 'resnet50'
        self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/pcb'
        self.OUTPUT_DIR = './output/pcb'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.LOSS_TYPE = 'pcb'
        self.TEST_WEIGHT = './output/resnet50_185.pth'
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]
        self.DEVICE_ID = "cuda:03"

class Config6(DefaultConfig):
    """
    Config for pcb+triplet
    mAP: 63.8% , Rank1: 81.1%, Rank5: 91.4% , Rank10:93.7%  @epoch: 200
    """

    def __init__(self):
        super(Config6, self).__init__()
        self.CFG_NAME = 'pcb_triplet'
        self.MODEL_NAME = 'resnet50'
        self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/pcb_triplet'
        self.OUTPUT_DIR = './output/pcb_triplet'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.LOSS_TYPE = 'pcb+triplet'
        self.TEST_WEIGHT = './output/resnet50_185.pth'
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]

class Config7(DefaultConfig):
    """
    Config for aligned algorithm: global+local, last stride then downsample 
    mAP:81.7% , Rank1: 91.2%, Rank5:96.7%, Rank10:98.1%, @epoch 
    """

    def __init__(self):
        super(Config7, self).__init__()
        self.CFG_NAME = 'aligned_global_local'
        self.MODEL_NAME = 'resnet50'
        self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/aligned_global_local'
        self.OUTPUT_DIR = './output/aligned_global_local'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.LOSS_TYPE = 'aligned'
        self.TEST_WEIGHT = './output/resnet50_185.pth'
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]

class Config8(DefaultConfig):
    """
    Config for aligned algorithm: global+local, 
    No last stride then downsample 
    mAP , Rank1 , @epoch 
    """

    def __init__(self):
        super(Config8, self).__init__()
        self.CFG_NAME = 'Original aligned global local'
        self.MODEL_NAME = 'resnet50'
        self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/aligned_global_local_no_last_stride'
        self.OUTPUT_DIR = './output/aligned_global_local_no_last_stride'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.LOSS_TYPE = 'aligned'
        self.TEST_WEIGHT = './output/resnet50_185.pth'
        self.LAST_STRIDE = 2
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]

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
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.LOSS_TYPE = 'aligned+pcb'
        self.TEST_WEIGHT = './output/aligned_pcb/resnet50_175.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'euclidean'

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
        self.TEST_WEIGHT = './output/aligned_pcb_center/resnet50_80.pth'
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
        self.TEST_WEIGHT = './output/aligned_pcb_center_DukeMTMCREID/resnet50_100.pth'
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
        self.TEST_WEIGHT = './output/aligned_pcb_DukeMTMCREID/resnet50_80.pth'
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
        self.TEST_WEIGHT = './output/aligned_pcb_CUHK03/resnet50_150.pth'
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
        self.TEST_WEIGHT = './output/aligned_pcb_center_CUHK03/resnet50_80.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = True
        self.INPUT_SIZE = [256, 128]
        self.TEST_IMS_PER_BATCH = 32
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'euclidean'

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
        self.LOG_DIR = './log/aligned_pcb_adapt_optimizer'
        self.OUTPUT_DIR = './output/aligned_pcb_adapt_optimizer'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.LOSS_TYPE = 'aligned+pcb'
        self.TEST_WEIGHT = './output/aligned_pcb_adapt_optimizer/resnet50_175.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'euclidean'

class Config_9_2(DefaultConfig):
    """
    Config for aligned algorithm combine PCB: global+local, 
    No last stride then downsample 
    Test:
    Global Distance: mAP:79.2% , Rank1:91.2%  Rank5:96.5%, Rank10: 97.9%, @epoch:200
    Global Distance + Local Distance: mAP:80.6% , Rank1:91.5%  Rank5:97.1%, Rank10: 98.1%, @epoch:200
    """

    def __init__(self):
        super(Config_9_2, self).__init__()
        self.CFG_NAME = 'Aligned algorithm combine PCB with adapt optimizer SGD'
        self.MODEL_NAME = 'resnet50'
        self.DATA_DIR = '/home/vietthang/dataset/Market-1501-v15.09.15'
        self.LOG_DIR = './log/aligned_pcb_adapt_optimizer_SGD'
        self.OUTPUT_DIR = './output/aligned_pcb_adapt_optimizer_SGD'
        self.PRETRAIN_CHOICE = 'imagenet'
        self.PRETRAIN_PATH = './pretrained/resnet50-19c8e357.pth'
        self.BATCH_SIZE = 128
        self.CHECKPOINT_PERIOD = 5
        self.OPTIMIZER = 'SGD'
        self.LOSS_TYPE = 'aligned+pcb'
        self.TEST_WEIGHT = './output/aligned_pcb_adapt_optimizer_SGD/resnet50_175.pth'
        self.LAST_STRIDE = 1
        self.FLIP_FEATS = 'off'
        self.HARD_FACTOR = 0.2
        self.RERANKING = False
        self.INPUT_SIZE = [256, 128]
        self.TEST_DISTANCE = 'global_local'
        self.TEST_METHOD = 'euclidean'