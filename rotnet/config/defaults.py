from yacs.config import CfgNode as CN

_C = CN()

# ---------------------------------------------------------------------------- #
# Backbone
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.NAME = 'mobilenet_v2'
_C.MODEL.IN_FEATURES = 1
_C.MODEL.NUM_CLASSES = 360
_C.MODEL.PRETRAINED = True

# ---------------------------------------------------------------------------- #
# Criterion
# ---------------------------------------------------------------------------- #
_C.CRITERION = CN()
_C.CRITERION.NAME = 'crossentropy'

# ---------------------------------------------------------------------------- #
# Criterion
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
_C.LR_SCHEDULER.EPOCHES = 10
_C.LR_SCHEDULER.NAME = 'step_lr'
# for SteLR
_C.LR_SCHEDULER.STEP_SIZE = 3
