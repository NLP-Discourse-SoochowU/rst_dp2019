# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date: 2018/4/3
@Description:  parser_model
"""
from parser_model.rst_tree import rst_tree
import torch
import copy
import random
import numpy as np
import progressbar
from config import *
from utils.file_util import *
from parser_model.tree_obj import tree_obj
p = progressbar.ProgressBar()
torch.manual_seed(SEED)
random.seed(SEED)


class Parser:
    def __init__(self, model=None, tran_error_model=None, span_model=None):
        # load the spinn model
        self.spinn_model = model
        self.tran_error_model = tran_error_model
        self.span_model = span_model  # 实验发现调模型的时候span模型存在较大的"时差"
        self.tmp_edu = 0
        # 加载label信息
        self.labels_all = load_data(Action2ids_path)

    def restore(self, folder):
        self.spinn_model = load_data(os.path.join(folder, "model.pickle"))
        self.spinn_model.model = load_data(os.path.join(folder, "torch.bin"))
        return self.spinn_model

    def parsing_all(self, trees_eval, parse_model=None, tran_error_model=None):
        """ 用当前训练好的模型对评测树进行分析和构建树
        """
        if parse_model is not None:
            self.spinn_model = parse_model
        if tran_error_model is not None:
            self.tran_error_model = tran_error_model
        parsed_tree_list = []
        p.start(len(trees_eval))
        pro_idx = 1
        for tree in trees_eval:
            p.update(pro_idx)
            pro_idx += 1
            # 涉及树的深拷贝问题，速度很慢
            parsed_tree = self.parsing(tree)
            parsed_tree_list.append(parsed_tree)
        p.finish()
        return parsed_tree_list

    def parsing(self, tree):
        """ Desc: parsing tree(tree_obj) of a given discourse.
        """
        session = self.spinn_model.new_session(tree)
        stack_ = [None, None]
        queue_ = tree.edus[:]
        queue_.append(None)
        while not self.parsing_end(session):
            predict_score = self.spinn_model.score_tran(session)
            transition = self.choose_transition(predict_score, queue_, stack_)
            # print(transition, "buffer: ", len(queue_), "stack: ", len(stack_))
            if transition == SHIFT:
                rel, nucl = None, None
                transition_composition = transition
            else:
                if TRAIN_NR:
                    nr_score = self.spinn_model.score_nr(session)
                    nucl, rel = self.choose_nr(nr_score)
                else:
                    n_score = self.spinn_model.score_nucl(session)
                    nucl = self.choose_nucl(n_score)
                    r_score = self.spinn_model.score_rel(session)
                    if USE_NR_MASK:
                        mask_ = torch.Tensor(nr_mask[nucl2ids[nucl]]).view(1, -1)
                        r_score = mask_ * r_score
                    rel = self.choose_rel(r_score)
                transition_composition = (transition + "-" + nucl, rel)
            # form_tree
            session, angle_prop_all = self.spinn_model(session, transition_composition)
            # 上面在构建过程中，如果是reduce则将权重分配情况进行存储
            self.form_tree(stack_=stack_, queue_=queue_, transition=transition, child_ns_rel=nucl, rel=rel,
                           angle_prop_all=angle_prop_all)
        tree_ = stack_[-1]
        tree_.type = "Root"
        tree_.rel = "span"
        tree_.file_name = tree.file_name
        tree_.config_nodes(tree_)
        return tree_

    def two_step_parsing(self, tree):
        """ 实现分步骤解析，默认句子级别的先解析，篇章级别的后解析，看看北大的水有多深
            先对各个子句做一个解析，然后将各个句子的编码结果存储在buffer里面，然后以编码结果作为新的EDU的编码结果进行树的构建
        """
        sub_trees = tree.sents_edus
        doc_tree = tree_obj()
        for sub_tree in sub_trees:
            doc_tree.edus.append(self.parse_sub_tree(sub_tree))  # 得到的是各个句子的内部形状，还要再对上层单独进行工作
        return self.parse_sub_tree(doc_tree)

    def parse_sub_tree(self, sub_tree):
        """ 根据tree_obj一个单独的树（或者子树，获取根节点）
        """
        session = self.spinn_model.new_session(sub_tree)
        stack_sub = [None, None]
        queue_sub = sub_tree.edus[:]
        queue_sub.append(None)
        while not self.parsing_end(session):
            predict_score = self.spinn_model.score_tran(session)
            transition = self.choose_transition(predict_score, queue_sub, stack_sub)
            if transition == SHIFT:
                rel, nucl = None, None
            else:
                rel_score = self.spinn_model.score_rel(session)
                nucl_score = self.spinn_model.score_nucl(session)
                rel = self.choose_rel(rel_score)
                nucl = self.choose_nucl(nucl_score)
            # form_tree
            self.form_tree(stack_=stack_sub, queue_=queue_sub, transition=transition, child_ns_rel=nucl, rel=rel)
            session = self.spinn_model(session, transition)
        return stack_sub[-1]

    def parse(self, parse_model, parse_data):
        """ :param parse_model: 训练好的模型
            :param parse_data: 测试数据集
        """
        safe_mkdir(TREES_PARSED)
        trees_eval_ = parse_data[:]
        trees_list = self.parsing_all(trees_eval_, parse_model)
        print("解析成功!")
        # 画树
        strtree_dict = dict()
        for tree in trees_list:
            strtree, draw_file = self.draw_one_tree(tree, TREES_PARSED)
            strtree_dict[draw_file] = strtree

        # 存储解析得到的树和对应的字符串化的树
        save_data(trees_list, TREES_PARSED + "/trees_list.pkl")
        save_data(strtree_dict, TREES_PARSED + "/strtrees_dict.pkl")

    def draw_one_tree(self, tree, path):
        """ 根据给定的rst树对象画出图像返回并存储到对应路径
        """
        self.tmp_edu = 1
        strtree = self.parse2strtree(tree)
        draw_file = path + "/" + tree.file_name.split(".")[0] + ".ps"
        return strtree, draw_file

    def parse2strtree(self, root):
        """
         ( NN-textualorganization ( SN-attribution ( EDU 1 )  ( NN-list ( EDU 2 )  ( EDU 3 )  )  )  ( EDU 5 )  )

        括号和括号之间两个空格，其余位置1个空格
        """
        if root.left_child is None:
            tmp_str = "( EDU " + str(self.tmp_edu) + " )"
            self.tmp_edu += 1
            return tmp_str
        else:
            tmp_str = ("( " + root.child_NS_rel) + "-" + root.child_rel + " " + self.parse2strtree(root.left_child) + \
                      "  " + self.parse2strtree(root.right_child) + "  )"
            return tmp_str

    @staticmethod
    def form_tree(stack_, queue_, transition, child_ns_rel, rel, angle_prop_all=None):
        type_dict = {"N": "Nucleus", "S": "Satellite"}
        if transition == SHIFT:
            edu_tmp = queue_.pop(0)
            stack_.append(edu_tmp)
        else:
            child_rel = rel
            # 建树
            right_c = stack_.pop()
            left_c = stack_.pop()
            left_c.type = type_dict[child_ns_rel[0]]
            right_c.type = type_dict[child_ns_rel[1]]
            if child_ns_rel == "NN":
                left_c.rel = child_rel
                right_c.rel = child_rel
            else:
                left_c.rel = child_rel if left_c.type == "Satellite" else "span"
                right_c.rel = child_rel if right_c.type == "Satellite" else "span"
            new_tree_node = rst_tree(l_ch=left_c, r_ch=right_c, ch_ns_rel=child_ns_rel, child_rel=child_rel,
                                     temp_edus_count=left_c.temp_edus_count + right_c.temp_edus_count)
            new_tree_node.attn_assigned = angle_prop_all  # attention分配
            stack_.append(new_tree_node)

    @staticmethod
    def clone_qs(q_s):
        """ 输入的是queue和stack的信息，进行拷贝
        """
        q_s_ = [copy.copy(edu) for edu in q_s]
        return q_s_

    @staticmethod
    def parsing_end(session):
        stack_, buffer_, _ = session
        state_ = True if len(stack_) == 3 and len(buffer_) == 1 else False
        return state_

    @staticmethod
    def get_random_reduce(type_):
        if type_ == "ns":
            action_ids = random.randint(1, Transition_num - 1)
        else:
            action_ids = None
            input("关系集合有待扩充")
        return ids2action[action_ids]

    @staticmethod
    def choose_transition(score, queue_, stack_):
        """
        找最大分数，专注于解决转移操作的问题
        """
        score = score.data.numpy()[0]
        transition_idx = np.where(score == np.max(score))[0][0]
        transition = ids2action[transition_idx]
        if transition_idx == 0 and len(queue_) < 2:
            # shift：满足当前预测结果为shift并且buffer中存在足够的有效数据
            transition = REDUCE
        if transition_idx == 1 and len(stack_) < 4:
            # reduce：预测得到的结果为reduce但是stack中不存在足够的数据供reduce，注意最底层两个是备用数据
            transition = SHIFT
        return transition

    @staticmethod
    def choose_nucl(score_nucl):
        score_nucl = score_nucl.data.numpy()[0]
        nucl_idx = np.where(score_nucl == np.max(score_nucl))[0][0]
        nucl = ids2nucl[nucl_idx]
        return nucl

    @staticmethod
    def choose_rel(score_rel):
        """ 假设空间作为一个绝对强烈的规则，但是强烈的规则会使得预测分数降低，因为评测只看 span + rel，F看 ns_rel
        """
        score_rel = score_rel.data.numpy()[0]
        rel_idx = np.where(score_rel == np.max(score_rel))[0][0]
        rel = ids2coarse[rel_idx]
        return rel

    @staticmethod
    def choose_nr(score_nr):
        score_nr = score_nr.data.numpy()[0]
        nr_idx = np.where(score_nr == np.max(score_nr))[0][0]
        nucl, rel = ids2nr[nr_idx].split("-")[0], "-".join(ids2nr[nr_idx].split("-")[1:])
        return nucl, rel
