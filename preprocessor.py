# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description:
"""
import numpy as np
from path_config import *
from config import *
from config import *
from utils.file_util import *
from utils.text_process_util import rm_edge_s, get_sent_words_syns, s_list


def build_glove_word_dict():
    """ 对词汇到文件名的映射，然后对向量的映射 创建word2ids和ids2vec
        对词频信息的存储分析
    """
    # 遍历edus文件并存储edus文本数据
    train_edus_token_list = build_rst_edus_tokens("train")
    test_edus_token_list = build_rst_edus_tokens("test")
    edus_token_list = train_edus_token_list.copy()
    edus_token_list.extend(test_edus_token_list.copy())

    # glove words
    words_set = set()
    with open(GLOVE_PATH, "r") as f:
        for line in f:
            tokens = line.split()
            words_set.add(tokens[0])
    # key dictionary
    word2ids, pos2ids, word2freq = dict(), dict(), dict()
    # 对 <UNK> 和 <PAD> 部分进行添加
    word2ids[PAD], word2ids[UNK], idx = 0, 1, 2
    for edu_tokens in edus_token_list:
        for token in edu_tokens:
            token = token.lower()
            if token not in word2freq.keys():
                word2freq[token] = 1
            elif token not in word2ids.keys() and token in words_set:
                word2freq[token] += 1
                word2ids[token] = idx
                idx += 1
            else:
                word2freq[token] += 1
    # 低频词过滤
    for word in word2freq.keys():
        if word not in word2ids.keys():
            word2ids[word] = word2ids[UNK]
    save_data(word2ids, VOC_WORD2IDS_PATH)  # 词汇到下标的映射
    ids2vec = dict()
    with open(GLOVE_PATH, "r") as f:
        for line in f:
            tokens = line.split()
            word = tokens[0]
            vec = np.array([[float(token) for token in tokens[1:]]])
            if tokens[0] in word2ids.keys() and word2ids[tokens[0]] != UNK_ids:
                ids2vec[word2ids[word]] = vec
    # transform into numpy array
    # PAD and UNK
    embed = [np.zeros(shape=(EMBED_SIZE,), dtype=np.float32)]
    embed = np.append(embed, [np.random.uniform(-0.25, 0.25, EMBED_SIZE)], axis=0)
    # others
    idx_valid = list(ids2vec.keys())
    idx_valid.sort()
    for idx in idx_valid:
        embed = np.append(embed, ids2vec[idx], axis=0)
    save_data(embed, VOC_VEC_PATH)  # 词向量

    # 将获取的edus_list转换成eduids_list
    train_ids_list = edus_toks2ids(train_edus_token_list, word2ids)
    test_ids_list = edus_toks2ids(test_edus_token_list, word2ids)
    # 存储
    write_iterate(word2ids.keys(), WORDS_PATH)
    save_data(train_ids_list, RST_TRAIN_EDUS_IDS_PATH)
    save_data(test_ids_list, RST_TEST_EDUS_IDS_PATH)
    # save_data(ids2freq, WORD_FREQUENCY)


def build_rst_edus_tokens(type_):
    """ train or test
    """
    edus_token_list = []
    txt_path_ = RST_DT_TRAIN_PATH if type_ == "train" else RST_DT_TEST_PATH
    for filename in os.listdir(txt_path_):
        if filename.endswith(".out.edus"):
            tmp_edu_file = os.path.join(txt_path_, filename)
            tmp_sent_file = tmp_edu_file.replace(".out.edus", ".out")
            tmp_edus_token_list = get_edus_tokens_list(tmp_edu_file, tmp_sent_file)
            edus_token_list.extend(tmp_edus_token_list)
    return edus_token_list


def get_edus_tokens_list(edu_path, sentence_path):
    edus_list, sent_list = [], []
    with open(sentence_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            sent_list.append(line)

    with open(edu_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            edus_list.append(line)
    # convert2tokens
    temp_sent_ids, edus_token_list, temp_edus_token_list = 0, [], []
    flag = True
    for edu in edus_list:
        edu_tokens = get_sent_words_syns(edu)
        for idx in range(len(edu_tokens)):
            if edu_tokens[idx] == "\"":
                edu_tokens[idx] = "``" if flag else "''"
                flag = not flag
        edus_token_list.append(edu_tokens)
    return edus_token_list


def edus_toks2ids(edu_toks_list, word2ids):
    """ 将训练cbos的论元句子们转换成ids序列， 将训练cdtb论元关系的论元对转成对应的论元对的tuple ids 列表并返回
    """
    tok_list_ids = []
    for line in edu_toks_list:
        line_ids = get_line_ids(toks=line, word2ids=word2ids)
        tok_list_ids.append(line_ids)
    # 数据存储
    return tok_list_ids


def get_line_ids(toks, word2ids=None):
    """ 根据输入的行信息和word2ids字典对句子进行转换, no padding
    """
    if word2ids is None:
        input("rst_utils.py")
    line_ids = []
    for tok in toks:
        tok = tok.lower()
        if tok in word2ids.keys():
            line_ids.append(word2ids[tok])
        else:
            line_ids.append(word2ids[UNK])
    return line_ids


if __name__ == "__main__":
    # build voc (word2ids, pos2ids, syn2ids, tag2ids)
    # data_sets = load_data(DATA_SETS_RAW)
    # build_voc((data_sets[0], data_sets[1]))
    train_path = "data/rst_dt/TRAINING_GROUPED/group_7/train/file5.out"
    train_path2 = "data/rst_dt/TRAINING_GROUPED/group_7/train/file5"
    with open(train_path, "r") as f:
        lines = f.readlines()
    sent_b = False
    sent_ = ""
    new_lines = []
    for line in lines:
        if line[0] == " ":
            if sent_ != "":
                new_lines.append(sent_)
            sent_ = line.strip()
        else:
            sent_ += " " + line.strip()
    if sent_ != "":
        new_lines.append(sent_)
    write_iterate(new_lines, train_path2)
