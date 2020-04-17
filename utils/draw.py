# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date: 2018.5.23
@Description:
"""
from config import *
from parser_model.parser import Parser
from utils.file_util import *
from parser_model.form_data import form_data
from nltk.draw.util import CanvasFrame, TextWidget
from nltk.draw import TreeWidget
from nltk import Tree


def draw_gold_test():
    """
    从文件读取数据创建rst_tree对象
    :return:
    """
    form_data_o = form_data()
    parser = Parser()
    for file_name in os.listdir(RST_DT_TEST_PATH):
        if file_name.endswith('.out.dis'):
            root = form_data_o.build_one_tree(RST_DT_TEST_PATH, file_name)
            parser.tmp_edu = 1
            strtree = parser.parse2strtree(root)
            save_path = TREES_PARSED + "/gold_test_trees/" + root.file_name.split(".")[0] + ".ps"
            draw_one_tree(strtree, save_path)


def draw_all_parsed():
    """
    将生文本画出树
    :return:
    """
    strtrees_dict_path = TREES_PARSED + "/strtrees_dict.pkl"
    strtrees_dict = load_data(strtrees_dict_path)
    for draw_file in strtrees_dict.keys():
        strtree = strtrees_dict[draw_file]
        draw_one_tree(strtree, draw_file)


def draw_one_tree(strtree, draw_file):
    cf = CanvasFrame()
    t = Tree.fromstring(strtree)
    tc = TreeWidget(cf.canvas(), t, draggable=1)
    cf.add_widget(tc, 1200, 0)  # (10,10) offsets
    # edus 文本
    edus_txt = ""
    c = cf.canvas()
    edu_path = RAW_TXT + "/" + draw_file.split("/")[2].split(".")[0] + ".out.edu"
    with open(edu_path, "r") as f:
        for line in f:
            edus_txt += line
    edus_txt = TextWidget(c, edus_txt, draggable=1)
    cf.add_widget(edus_txt, 1400, 0)
    user_choice = input("直接打印(a) or 存到文件(b): ")
    if user_choice == "a":
        cf.mainloop()
    else:
        cf.print_to_file(draw_file)
        cf.destroy()
