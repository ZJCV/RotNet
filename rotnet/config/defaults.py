from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------- #
# Train
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.NAME = 'RotNet.train'
_C.TRAIN.MAX_ITER = 10000
_C.TRAIN.LOG_STEP = 10
_C.TRAIN.SAVE_STEP = 2500
_C.TRAIN.EVAL_STEP = 2500

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.NAME = 'RotNet.test'

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.NAME = 'mobilenet_v2'
_C.MODEL.INPUT_SIZE = (224, 224)
_C.MODEL.IN_FEATURES = 1
_C.MODEL.NUM_CLASSES = 360
_C.MODEL.PRETRAINED = True

# ---------------------------------------------------------------------------- #
# Criterion
# ---------------------------------------------------------------------------- #
_C.CRITERION = CN()
_C.CRITERION.NAME = 'crossentropy'

# ---------------------------------------------------------------------------- #
# Optimizer
# ---------------------------------------------------------------------------- #
_C.OPTIMIZER = CN()
_C.OPTIMIZER.NAME = 'sgd'
_C.OPTIMIZER.LR = 1e-3
_C.OPTIMIZER.WEIGHT_DECAY = 3e-4
# for sgd
_C.OPTIMIZER.MOMENTUM = 0.9

# ---------------------------------------------------------------------------- #
# LR_Scheduler
# ---------------------------------------------------------------------------- #
_C.LR_SCHEDULER = CN()
_C.LR_SCHEDULER.NAME = 'step_lr'
_C.LR_SCHEDULER.GAMMA = 0.1
# for SteLR
_C.LR_SCHEDULER.STEP_SIZE = 400
# for MultiStepLR
_C.LR_SCHEDULER.MILESTONES = [2500, 6000]

# ---------------------------------------------------------------------------- #
# DataSets
# ---------------------------------------------------------------------------- #
_C.DATASETS = CN()
_C.DATASETS.TRAIN = ['FashionMNIST']
_C.DATASETS.TEST = ['FashionMNIST']

# ---------------------------------------------------------------------------- #
# DataLoader
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.TRAIN_BATCH_SIZE = 128
_C.DATALOADER.TEST_BATCH_SIZE = 10
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Output
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.DIR = 'outputs/mobilenetv2'
