# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date:
@Description:
"""
import torch
import os
import discoseg.buildedu as buildedu
from config import *
from parser_model.model import SPINN
from parser_model.parser import Parser
from utils.file_util import write_iterate
from parser_model.form_data import Builder


class parse:
    def __init__(self):
        self.model = self.load_model()

    def segmentation(self):
        # 对生文本的树形构建 segmentation
        buildedu.main(seg_model_path, seg_voc_path, RAW_TXT, RAW_TXT)
        # 生成EDU文件并为各个目标构建tree_obj对象方便调用
        self.form_edus()

    def parse(self):
        self.segmentation()
        # 生成 tree_obj对象
        builder = Builder()
        if not LOAD_TEST:
            parse_trees = builder.build_tree_obj_list()
        else:
            parse_trees = load_data(RST_TEST_TREES)
        parse_model = Parser()
        parse_model.parse(parse_model=self.model, parse_data=parse_trees)

    @staticmethod
    def load_model():
        print("loading...")
        word2ids = load_data(VOC_WORD2IDS_PATH)
        pos2ids = load_data(POS_word2ids_PATH)
        word_embed = load_data(VOC_VEC_PATH)
        model = SPINN(word2ids, pos2ids, word_embed)
        model.load_state_dict(torch.load(PRETRAINED))
        print("The model has been loaded successfully!")
        return model

    @staticmethod
    def form_edus():
        """
            对segmentation得到的文件阅读并对tree_obj进行构建
            :return:
        """
        for file_name in os.listdir(RAW_TXT):
            if file_name.endswith(".out.merge"):
                tmp_edu_id = 1
                tmp_edu = ""
                tmp_file_edus = []
                path_ = os.path.join(RAW_TXT, file_name)
                with open(path_, "r") as f:
                    edu_path_ = path_.replace(".merge", ".edu")
                    if os.path.exists(edu_path_):
                        print("已经进行过edu切割！")
                        return
                    for line in f:
                        tmp_line = line.strip()
                        if len(tmp_line) == 0:
                            continue
                        line_split = tmp_line.split()
                        if line_split[-1] == str(tmp_edu_id):
                            tmp_edu += (line_split[2] + " ")
                        else:
                            tmp_file_edus.append(tmp_edu.strip())
                            tmp_edu_id += 1
                            tmp_edu = line_split[2] + " "
                    tmp_file_edus.append(tmp_edu.strip())
                    write_iterate(tmp_file_edus, edu_path_)


def judge_seg():
    """ 判断篇章是否已经进行切割
    """
    flag = True
    for file_name in os.listdir(RAW_TXT):
        if file_name.endswith(".out"):
            path_ = os.path.join(RAW_TXT, file_name)
            if not os.path.exists(path_ + ".merge"):
                flag = False
                break
    return flag


if __name__ == "__main__":
    print("请将要解析的篇章放置在data/raw_txt目录下！")
    if not judge_seg():
        print("请先调用corpus_rst.sh对raw_txt下的篇章预处理！")
    else:
        parse_ = parse()
        parse_.parse()
        print("解析到的篇章树rst_tree对象以列表形式存放于：" + TREES_PARSED + "/trees_list.pkl")
