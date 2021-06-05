from yacs.config import CfgNode as CN

# config definition
_C = CN()

_C.INPUT_SIZE = 320
_C.SCALE = 2

_C.SEED = 42

_C.dist_url = 'env://'
_C.world_size = 1

# dataset config
_C.DATASET = CN()
_C.DATASET.ROOT = '/home/jc3/Data'  # the root of dataset
_C.DATASET.CHALLENGE = 'singlecoil'  # the task of ours, singlecoil or multicoil
_C.DATASET.MODE = ''  # train or test

_C.TRANSFORMS = CN()
_C.TRANSFORMS.MASKTYPE = 'random'  # "random" or "equispaced"
_C.TRANSFORMS.CENTER_FRACTIONS = [0.08]
_C.TRANSFORMS.ACCELERATIONS = [4]

# model config
_C.MODEL = CN()
_C.MODEL.INPUT_DIM = 2   # the channel of input
_C.MODEL.OUTPUT_DIM = 1   # the channel of output
_C.MODEL.HEAD_HIDDEN_DIM = 16  # the hidden dim of Head
_C.MODEL.TRANSFORMER_DEPTH = 4  # the depth of the transformer
_C.MODEL.TRANSFORMER_NUM_HEADS = 4  # the head's num of multi head attention
_C.MODEL.TRANSFORMER_MLP_RATIO = 3  # the MLP RATIO Of transformer
_C.MODEL.TRANSFORMER_EMBED_DIM = 256  # the EMBED DIM of transformer
_C.MODEL.PATCH_SIZE = 16

_C.MODEL.CTDEPTH = 4

_C.MULTI = CN()

# the solver config

_C.SOLVER = CN()
_C.SOLVER.DEVICE = 'cuda'
_C.SOLVER.DEVICE_IDS = [0, 1]  # if [] use cpu, else gpu
_C.SOLVER.LR = 5e-5
_C.SOLVER.WEIGHT_DECAY = 1e-4
_C.SOLVER.LR_DROP = [40]
_C.SOLVER.BATCH_SIZE = 4
_C.SOLVER.NUM_WORKERS = 16
_C.SOLVER.PRINT_FREQ = 10

# the others config
_C.RESUME = ''  # model resume path
_C.OUTPUTDIR = './weights_reconstruction_multi_early'  # the model output dir

#the train configs
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 50  # the train epochs

_C.WORK_TYPE = 'reconstruction'

_C.NOISE_STD = 1e-5
_C.USE_CL1_LOSS = False
_C.USE_MULTI_MODEL = True


