# -*- coding: utf-8 -*-

"""
@Author: lemons
@Date:
@Description: 针对rst tree的处理，分别得到句中和句子间的信息
"""

def form_sent_trees(rst_trees):
    """
    传入若干树，每棵树均存储了文件名，根据文件名找到句子级文档信息
    后根遍历树，如果当前结点的文本部分依次和句子相同，进行子树分割，同时将当前部分设置为sent节点（句子作为叶子）
    :param rst_trees:
    :return:一系列句子树以及若干以句子为叶子的篇章树
    """
    ...
