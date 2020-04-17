# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date:
@Description: Global Configuration
"""
from path_config import *
from utils.file_util import load_data

# ================================== Experimental ==================================
VERSION = 7
SET_of_version = "458_No_feat"
AREA_REL = True
AREA_NUCL = True
AREA_TRAN = False
MLP_LAYERS = 1
POS_EMBED_SIZE = 50  # 46
SPINN_HIDDEN = 512
USE_state_tracker = True
Tracking_With_GRU = False
TRAN_LOSS_VERSION = 2  # 1: origin 2: margin loss
NUCL_LOSS_VERSION = 2
REL_LOSS_VERSION = 1
GATE_ANGLE_NUM = 1

USE_NR_MASK = False  # rel prediction is dependent on nuclearity detection
TRAIN_NR = True
TRAIN_NUCL_CONSTRAINT = True
CONSTRAINT_LAMBDA = 0.6

USE_ELMo = True
LOAD_ELMo = False
CROSS_VAL = True
CROSS_GROUP_ID = 7
SAVE_MODEL = True
USE_SPAN_MODEL = False
USE_W_P_IN_BI_LSTM = True

# ================================== statistic ==================================
USE_POS = True
EMBED_SIZE = 1024 if USE_ELMo else 300
GLOVE_EMBED_SIZE = 300
CONN_EMBED_SIZE = 100  # 100 -> 64 -> 100
PAD_SIZE = 56  # for version using CNN
proj_dropout = 0.2
gate_drop_out_rate = 0.2
l2_penalty = 1e-5
SEED = 2
LEARNING_RATE_spinn = 0.001
BATCH_SIZE_spinn = 1
EPOCH_ALL = 20
SKIP_STEP_spinn = 16
SKIP_STEP_min = 4  # 最低跨度
SKIP_REDUCE_UNIT = 2
TRAIN_SET_SIZE = 347
SKIP_BOUNDARY = int((TRAIN_SET_SIZE * EPOCH_ALL * 0.8) / ((SKIP_STEP_spinn-SKIP_STEP_min)/SKIP_REDUCE_UNIT))
DESC = "DESC: Use LSTM as Encoder"
OVER = "===============  Over a batch  ==============="

# ================================== mlp ==================================
mlp_input_size = SPINN_HIDDEN
if USE_state_tracker:
    area_input_size = SPINN_HIDDEN * 5
else:
    area_input_size = SPINN_HIDDEN * 4
mlp_dropout = 0.2

# ================================== others ==================================
PAD = "<PAD>"
PAD_ids = 0
UNK = "<UNK>"
UNK_ids = 1
LOW_FREQ = 1
Transition_num = 2  # SHIFT or REDUCE
NUCL_NUM = 3  # NN NS SN
BEAM_SIZE = int(Transition_num / 2)
FINE_REL_NUM = 56
COARSE_REL_NUM = 18
NR_NUM = 42

# 操作标签 只关心核型
SHIFT = "SHIFT"
REDUCE = "REDUCE"
REDUCE_NN = "REDUCE-NN"
REDUCE_NS = "REDUCE-NS"
REDUCE_SN = "REDUCE-SN"
NN = "NN"
NS = "NS"
SN = "SN"
action2ids = {SHIFT: 0, REDUCE: 1}
ids2action = {0: SHIFT, 1: REDUCE}

nucl2ids = {NN: 0, NS: 1, SN: 2}
ids2nucl = {0: NN, 1: NS, 2: SN}
ns_dict = {"Satellite": 0, "Nucleus": 1, "Root": 2}
ns_dict_ = {0: "Satellite", 1: "Nucleus", 2: "Root"}

coarse2ids = load_data(REL_coarse2ids)
ids2coarse = load_data(REL_ids2coarse)

nr2ids = load_data(NR2ids_path)
ids2nr = load_data(Ids2nr_path)
nr_mask = load_data(NR_MASK_PATH)

LOAD_TEST = True
DEV_DISTRIBUTE = "data/distributions/group_" + str(CROSS_GROUP_ID) + "/" + SET_of_version + "_dev_distribution.pkl" if \
    CROSS_VAL else "data/distributions/" + SET_of_version + "_dev_distribution.pkl"
# TEST_DISTRIBUTE = "data/distributions/" + SET_of_version + "_test_distribution.pkl"

ALL_LABELS_NUM = 43
LABEL_EMBED_SIZE = 5
