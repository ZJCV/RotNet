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
