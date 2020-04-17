from parser.model_config import *
import numpy as np
from utils.file_util import *

# sents_ids_path = "data/pdtb_conv_train/sents_ids.pkl"
word2ids_path = "data/voc/word2ids.pkl"

# sents_ids.shape : (?, 200)
def generate_sample(pair_ids):  # 根据句子，返回以word ids为内容的一批数据
    for item in pair_ids:
        yield item[0], item[1], item[2], item[3]

def batch_gen(pair_ids_path):
    pair_ids = load_data(pair_ids_path)
    single_gen = generate_sample(pair_ids)
    while True:
        arg1_batch = np.zeros([BATCH_SIZE, PAD_SIZE], dtype=np.int32)
        arg2_batch = np.zeros([BATCH_SIZE, PAD_SIZE], dtype=np.int32)
        connective_batch = np.zeros([BATCH_SIZE, 1], dtype=np.int32)
        rel_batch = np.zeros([BATCH_SIZE, N_TAGS], dtype=np.int32)
        for index in range(BATCH_SIZE):
            arg1_batch[index], arg2_batch[index], connective_batch[index], rel_batch[index] = next(single_gen)
        yield arg1_batch, arg2_batch, connective_batch, rel_batch


