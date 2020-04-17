"""
统计当前各个text span包含edu个数最多是多少，存成set
"""
import numpy as np
from config import RST_TRAIN_TREES_ELMo_RST, RST_TEST_TREES_ELMo_RST, RST_TRAIN_ELMo_TREES, RST_TEST_ELMo_TREES
from utils.file_util import *

trees_list = load_data(RST_TRAIN_ELMo_TREES)
trees_list += load_data(RST_TEST_ELMo_TREES)
edunum2count = dict()
total_num = 0.
eva_num = 7
eva_count = 0.
for tree in trees_list:
    for node in tree.nodes:
        edu_count = node.temp_edus_count
        if edu_count == 1:
            continue
        total_num += 1
        if edu_count >= eva_num:
            eva_count += 1
        if edu_count in list(edunum2count.keys()):
            edunum2count[edu_count] += 1
        else:
            edunum2count[edu_count] = 1
keys_s = sorted(edunum2count.keys())
for key in keys_s:
    print(key, ": ", edunum2count[key], ", ")
print(eva_count / total_num)
print(eva_count, total_num)

