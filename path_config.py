# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date:
@Description:
"""

# Data.glove
GLOVE_PATH = "data/gloves"
GLOVE50_embedding_dir = "data/gloves/gloves_50"
GLOVE50_vec_dir = "data/gloves_50/embed.pkl"
GLOVE100_embedding_dir = "data/gloves/gloves_100"
GLOVE100_vec_dir = "data/gloves_100/embed.pkl"

# Data.voc & pos & so on 针对当前语料库的词库形成
WORDS_PATH = "data/voc/words.tsv"
VOC_WORD2IDS_PATH = "data/voc/word2ids.pkl"
VOC_VEC_PATH = "data/voc/ids2vec.pkl"
POS_word2ids_PATH = "data/voc/pos2ids.pkl"
POS_TAGS_PATH = "data/voc/pos_tags.tsv"

# Data.RST
RST_DT_TRAIN_PATH = "data/rst_dt/TRAINING_GROUPED/group_7/train"
RST_DT_DEV_PATH = "data/rst_dt/TRAINING_GROUPED/group_7/dev"
RST_DT_TEST_PATH = "data/rst_dt/TEST"

# 根据Glove生成
RST_TRAIN_TREES = "data/rst_dt/GloVe/train_trees.pkl"
RST_TEST_TREES = "data/rst_dt/GloVe/test_trees.pkl"
RST_DEV_TREES = "data/rst_dt/GloVe/dev_trees.pkl"
RST_TRAIN_TREES_RST = "data/rst_dt/GloVe/train_trees_rst.pkl"
RST_TEST_TREES_RST = "data/rst_dt/GloVe/test_trees_rst.pkl"
RST_DEV_TREES_RST = "data/rst_dt/GloVe/dev_trees_rst.pkl"

# 根据ELMo生成
RST_TRAIN_ELMo_TREES = "data/rst_dt/ELMo/train_trees.pkl"
RST_TEST_ELMo_TREES = "data/rst_dt/ELMo/test_trees.pkl"
RST_DEV_ELMo_TREES = "data/rst_dt/ELMo/dev_trees.pkl"
RST_TRAIN_TREES_ELMo_RST = "data/rst_dt/ELMo/train_trees_rst.pkl"
RST_TEST_TREES_ELMo_RST = "data/rst_dt/ELMo/test_trees_rst.pkl"
RST_DEV_TREES_ELMo_RST = "data/rst_dt/ELMo/dev_trees_rst.pkl"

RST_TRAIN_EDUS_IDS_PATH = "data/rst_dt/RST_EDUS/train_edus_ids.pkl"
RST_TEST_EDUS_IDS_PATH = "data/rst_dt/RST_EDUS/test_edus_ids.pkl"
RST_DEV_EDUS_IDS_PATH = "data/rst_dt/RST_EDUS/dev_edus_ids.pkl"

# 映射表
REL_raw2coarse = "data/rst_dt/map_tables/rel_raw2coarse.pkl"
REL_coarse2ids = "data/rst_dt/map_tables/coarse2ids.pkl"
REL_ids2coarse = "data/rst_dt/map_tables/ids2coarse.pkl"
Action2ids_path = "data/rst_dt/map_tables/action2ids.pkl"
Ids2action_path = "data/rst_dt/map_tables/ids2action.pkl"
Action_label_dict_path = "data/rst_dt/map_tables/action_label_dict.pkl"
NR2ids_path = "data/rst_dt/map_tables/nr2ids.pkl"
Ids2nr_path = "data/rst_dt/map_tables/ids2nr.pkl"
NR_MASK_PATH = "data/rst_dt/map_tables/nr_make.pkl"
WORD_FREQUENCY = "data/rst_dt/map_tables/ids2freq.pkl"
CONN_RAW_LIST = "data/connective/conn_pdtb_list.pkl"
CONN_word2ids = "data/connective/conn2ids.pkl"
CONN_WORD_File = "data/connective/conn_word.tsv"

# margin_loss_distribution
POS_LOSS_PATH = "data/positive_loss.pkl"
NEG_LOSS_PATH = "data/negative_loss.pkl"
MARGIN_VALUE_AREA_PATH = "data/margin_value.pkl"

WSJ_PATH = "data/wsj"
BRACKET_PATH = "data/bracket.tsv"
BRACKET_INNER_PATH = "data/bracket_inner.tsv"
LOG_DIR = "data/log_file"
MODELS2SAVE = "data/models_saved"

# Config.Segmentation
seg_model_path = "data/segmentation_model_pkl/model.pickle.gz"
seg_voc_path = "data/segmentation_model_pkl/vocab.pickle.gz"
PRETRAINED = "data/rst_dt/models_saved/parsing_model/rel_max_model.pth"
RAW_TXT = "data/raw_txt"
TREES_PARSED = "data/trees_parsed"
