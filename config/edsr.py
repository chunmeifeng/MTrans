from yacs.config import CfgNode as CN

# config definition
_C = CN()

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


# the solver config

_C.SOLVER = CN()
_C.SOLVER.DEVICE = 'cuda'
_C.SOLVER.LR = 1e-4
_C.SOLVER.WEIGHT_DECAY = 1e-4
_C.SOLVER.LR_DROP = 80
_C.SOLVER.BATCH_SIZE = 8
_C.SOLVER.NUM_WORKERS = 16
_C.SOLVER.PRINT_FREQ = 10

# the others config
_C.RESUME = ''  # model resume path
_C.OUTPUTDIR = './weights_edsr'  # the model output dir
_C.TEST_OUTPUTDIR = 'outputs/edsr'

#the train configs
_C.TRAIN = CN()
_C.TRAIN.EPOCHS = 50  # the train epochs

_C.WORK_TYPE = 'sr'
_C.USE_CL1_LOSS = False
_C.USE_MULTI_MODEL = False

_C.n_resblocks = 4
_C.n_feats = 64
_C.res_scale = 1
_C.n_colors = 1

