# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date: 2018/4/5
@Description:
    语料需要转换，转换前对需要结合fetures特征,所以能够根据当前树的状态，得到对应的doc对象才能获取对应的特征遍历过程
    中验证树建立的合法性，进行前序遍历
    1 希望建立的树和文件保持一致
    2 希望shift-reduce过程和configuration保持同步，shift-reduce过程和文件的后续遍历保持一致，验证。
"""
from utils.file_util import *
from config import *
import progressbar
from parser_model.rst_tree import rst_tree
from parser_model.tree_obj import tree_obj
from utils.rst_utils import get_edus_info
from stanfordcorenlp import StanfordCoreNLP

path_to_jar = 'stanford-corenlp-full-2018-02-27'


class Builder:
    def __init__(self):
        self.root = None
        # tree list
        self.trees_list = []
        self.trees_rst_list = []
        self.nlp = StanfordCoreNLP(path_to_jar)
        self.elmo = None
        if USE_ELMo and LOAD_ELMo:
            print("Loading EMLo (Checking network)...")
            from allennlp.modules.elmo import Elmo
            options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/" \
                           "elmo_2x4096_512_2048cnn_2xhighway_options.json"
            weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo" \
                          "_2x4096_512_2048cnn_2xhighway_weights.hdf5"
            self.elmo = Elmo(options_file, weight_file, 2, dropout=0)
            print("Load Done.")

    def form_trees_type_(self, type_=None, rst_dt_path=None, save_tree_path=None,
                         save_tree_rst_path=None, raw_rel=False):
        """ 根据type_构建对应的类型的 trees list.
        """
        trees_dict = dict()
        trees_dict["train"] = [RST_DT_TRAIN_PATH, RST_TRAIN_ELMo_TREES, RST_TRAIN_TREES_ELMo_RST]
        trees_dict["test"] = [RST_DT_TEST_PATH, RST_TEST_ELMo_TREES, RST_TEST_TREES_ELMo_RST]
        trees_dict["dev"] = [RST_DT_DEV_PATH, RST_DEV_ELMo_TREES, RST_DEV_TREES_ELMo_RST]
        if type_ is not None:
            rst_dt_path = trees_dict[type_][0]  # source
            save_tree_path = trees_dict[type_][1]
            save_tree_rst_path = trees_dict[type_][2]
        self.build_specific_trees(raw_rel, rst_dt_path, save_tree_path, save_tree_rst_path)

    def build_specific_trees(self, raw_rel=False, rst_dt_path=None, save_tree_path=None, save_tree_rst_path=None):
        """ 指定存储，代码共享
            :param raw_rel: 初始化的关系
            :param rst_dt_path: 要建树的原始文件所在目录 train or dev or test
            :param save_tree_path: 要存储的 tree_obj 类型的文件所在目录
            :param save_tree_rst_path:  要存储的 rst_tree 类型的文件所在目录
        """
        print("begin...")
        p = progressbar.ProgressBar()
        p.start(len(os.listdir(rst_dt_path)))
        pro_idx = 1
        self.trees_list = []
        self.trees_rst_list = []
        for file_name in os.listdir(rst_dt_path):
            p.update(pro_idx)
            pro_idx += 1
            if file_name.endswith('.out.dis'):
                self.root = self.build_one_tree(rst_dt_path, file_name, raw_rel=raw_rel)
                self.trees_rst_list.append(self.root)
                tree_obj_ = tree_obj(self.root)
                self.trees_list.append(tree_obj_)
        p.finish()
        # save trees
        save_data(self.trees_list, save_tree_path)
        save_data(self.trees_rst_list, save_tree_rst_path)

    def build_one_tree(self, rst_dt_path, file_name, raw_rel=False):
        """ build edu_list and edu_ids according to EDU files
            这里进行对关系类型转fine关系，再转coarse关系
        """
        # print(file_name)
        temp_path = os.path.join(rst_dt_path, file_name)
        # EMLo: edus_ids_list = (edu_num, word_num, id_len)
        edus_boundary_list, edus_list, edu_span_list, edus_ids_list, edus_tag_ids_list, edus_conns_list, \
            edu_headword_ids_list, edu_has_center_word_list, edus_emb_emlo_list = \
            get_edus_info(temp_path.replace(".dis", ".edus"), temp_path.replace(".out.dis", ".out"), nlp=self.nlp,
                          file_name=file_name, elmo=self.elmo)

        # 在此处根据一个篇章的edu_list构建edu_vector_list, 注意这里存储的是word_ids，
        # 所以对词频的获取直接进行获取即可, 注意UNK词的词频的平滑处理
        ids2freq = load_data(WORD_FREQUENCY)
        edus_word_freq_list = []
        # 这部分关于概率计算
        for edus_ids in edus_ids_list:
            edus_freq = []
            for word_ids in edus_ids:
                edus_freq.append(ids2freq[word_ids])
            edus_word_freq_list.append(edus_freq)

        dis_tree_obj = open(temp_path, 'r')
        lines_list = dis_tree_obj.readlines()
        # 加载关系转换映射表
        rel_raw2coarse = load_data(REL_raw2coarse)
        if raw_rel is False:
            root = rst_tree(type_="Root", lines_list=lines_list, temp_line=lines_list[0], file_name=file_name,
                            rel="span", rel_raw2coarse=rel_raw2coarse)
        else:
            root = rst_tree(type_="Root", lines_list=lines_list, temp_line=lines_list[0], file_name=file_name,
                            rel="span", rel_raw2coarse=rel_raw2coarse, raw_rel=raw_rel)
        root.create_tree(temp_line_num=1, p_node_=self.root)
        edu_node = []
        if USE_ELMo:
            root.config_edus(temp_node=self.root, temp_edu_list=edus_list, temp_edu_span_list=edu_span_list,
                             temp_eduids_list=edus_ids_list, edus_tag_ids_list=edus_tag_ids_list,
                             edu_node=edu_node, edus_conns_list=edus_conns_list,
                             edu_headword_ids_list=edu_headword_ids_list,
                             edu_has_center_word_list=edu_has_center_word_list,
                             total_edus_num=len(edus_ids_list), edus_boundary_list=edus_boundary_list,
                             temp_edu_freq_list=edus_word_freq_list, edu_emlo_emb_list=edus_emb_emlo_list)
        else:
            root.config_edus(temp_node=self.root, temp_edu_list=edus_list, temp_edu_span_list=edu_span_list,
                             temp_eduids_list=edus_ids_list, edus_tag_ids_list=edus_tag_ids_list,
                             edu_node=edu_node, edus_conns_list=edus_conns_list,
                             edu_headword_ids_list=edu_headword_ids_list,
                             edu_has_center_word_list=edu_has_center_word_list,
                             total_edus_num=len(edus_ids_list), edus_boundary_list=edus_boundary_list,
                             temp_edu_freq_list=edus_word_freq_list)
        return root

    def build_tree_obj_list(self):
        """ 为生文本创建tree_obj对象
        """
        parse_trees = []
        for file_name in os.listdir(RAW_TXT):
            if file_name.endswith(".out"):
                tmp_edus_list = []
                sent_path = os.path.join(RAW_TXT, file_name)
                edu_path = sent_path + ".edu"
                edus_boundary_list, edus_list, edu_span_list, edus_ids_list, edus_tag_ids_list, edus_conns_list, \
                    edu_headword_ids_list, edu_has_center_word_list = get_edus_info(edu_path, sent_path, nlp=self.nlp,
                                                                                    file_name=file_name)
                for _ in range(len(edus_list)):
                    tmp_edu = rst_tree()
                    tmp_edu.temp_edu = edus_list.pop(0)
                    tmp_edu.temp_edu_span = edu_span_list.pop(0)
                    tmp_edu.temp_edu_ids = edus_ids_list.pop(0)
                    tmp_edu.temp_pos_ids = edus_tag_ids_list.pop(0)
                    tmp_edu.temp_edu_conn_ids = edus_conns_list.pop(0)
                    tmp_edu.temp_edu_heads = edu_headword_ids_list.pop(0)  # 当前EDU所有tokens对应的head word的ids
                    tmp_edu.temp_edu_has_center_word = edu_has_center_word_list.pop(0)  # 当前EDU是否包含句子中心词
                    tmp_edu.edu_node_boundary = edus_boundary_list.pop(0)
                    tmp_edu.inner_sent = True
                    tmp_edus_list.append(tmp_edu)
                # 构建tree_obj对象
                tmp_tree_obj = tree_obj()
                tmp_tree_obj.file_name = file_name
                tmp_tree_obj.assign_edus(tmp_edus_list)
                parse_trees.append(tmp_tree_obj)
        return parse_trees
