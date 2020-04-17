# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date:
@Description:
"""
import torch
import random
import progressbar
from config import *
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import Variable as Var
from parser_model.spinn_model import SPINN
# from parser_model.metric import Metrics
from parser_model.standard_parseval import Metrics
from parser_model.parser import Parser
from utils.file_util import *
from parser_model.tree_obj import tree_obj


class Trainer:
    """ Training Container
    """
    def __init__(self, train_trees_path=None, dev_trees_path=None, test_trees_path=None, load_static=False):
        torch.manual_seed(SEED)
        self.count_write = 0
        self.skip_steps = SKIP_STEP_spinn
        self.skip_boundary = SKIP_BOUNDARY
        word2ids = load_data(VOC_WORD2IDS_PATH)
        pos2ids = load_data(POS_word2ids_PATH)
        word_embed = load_data(VOC_VEC_PATH)
        # build the objective of SPINN
        self.model = SPINN(word2ids, pos2ids, word_embed)
        if load_static:
            self.model.load_state_dict(torch.load(PRETRAINED))
        # self.metric = Metrics()
        self.parser = Parser()
        self.zero_var = Var(torch.tensor(0.))
        # 数据加载
        self.train_trees_path = train_trees_path
        self.train_trees = load_data(train_trees_path)  # 因train_trees不经历parsing，所以不会打乱数据，不需要重新加载
        self.dev_trees_path = dev_trees_path
        self.dev_trees = load_data(dev_trees_path)
        self.test_trees_path = test_trees_path
        self.test_trees = load_data(test_trees_path)
        self.log_file = os.path.join(LOG_DIR, str(VERSION) + "-" + str(SET_of_version) + "log.txt")
        self.metric = Metrics(self.log_file)

    def session(self, tree):
        """ 针对当前 tree_obj 创建 session
        """
        session = self.model.new_session(tree)
        return session

    @staticmethod
    def tran_rel_parser(transition_rel):
        """ 分析拆解成transition和relation标签, 如果使用ns还要拆解出ns信息
        """
        transition = transition_rel if transition_rel == SHIFT else transition_rel[0]
        rel = None if transition_rel == SHIFT else transition_rel[1]
        tran, nucl = (transition, None) if transition == SHIFT else (transition.split("-")[0], transition.split("-")[1])
        return tran, nucl, rel

    @staticmethod
    def concat_torch(data_formed, data2concat):
        """ 将新数据和原数据拼接
        """
        if data_formed is None:
            data_formed = data2concat
        else:
            data_formed = torch.cat((data_formed, data2concat), 0)
        return data_formed

    def train_(self, test_desc, group_name=None):
        """ Training procedure, if we use cross loss instead of bce, we need to change the shape of logits
            BCE_loss是1对1计算，Cross是logits维度和one_hot标签维度一致，和标签种类一致.有待更新
        """
        p = progressbar.ProgressBar()
        random.seed(SEED)
        loss_func_tran = nn.MultiMarginLoss()
        loss_func_nucl = nn.MultiMarginLoss()
        loss_func_rel = nn.CrossEntropyLoss()
        constraint_loss = nn.CrossEntropyLoss()  # 对Nucl和rel添加限制
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE_spinn, weight_decay=l2_penalty)
        iter_count = batch_count = 0
        loss_tran = loss_nucl = loss_rel = loss_nr = 0.
        p.start(self.skip_steps)
        pro_idx = 1
        for epoch in range(EPOCH_ALL):
            if (epoch + 1) % 11 == 0:
                input("请查看epoch10的结果")
            random.shuffle(self.train_trees)
            for tree in self.train_trees:
                self.model.train()
                iter_count += 1
                session_gold = self.session(tree)
                tran_scores, nucl_scores, rel_scores, nr_scores = None, None, None, None
                tran_labels, nucl_labels, rel_labels, nr_labels = [], [], [], []
                for transition_rel in self.oracle(tree):
                    # 对state构建特征数据
                    tran, nucl, rel = self.tran_rel_parser(transition_rel)  # 解析
                    # 1. nucl and rel: scores and labels 这部分根据标准的状态信息进行计算即可维护两个状态转移
                    if rel is not None:
                        if TRAIN_NR:
                            tmp_nr_score = self.model.score_nr(session_gold)
                            nr_scores = self.concat_torch(nr_scores, tmp_nr_score)
                            nr_labels.append(nr2ids[nucl + "-" + rel])
                            if TRAIN_NUCL_CONSTRAINT:
                                tmp_nucl_score = self.model.score_nucl(session_gold)
                                nucl_scores = self.concat_torch(nucl_scores, tmp_nucl_score)
                                nucl_labels.append(nucl2ids[nucl])
                        else:
                            tmp_nucl_score = self.model.score_nucl(session_gold)
                            nucl_scores = self.concat_torch(nucl_scores, tmp_nucl_score)
                            nucl_labels.append(nucl2ids[nucl])
                            tmp_rel_score = self.model.score_rel(session_gold)
                            rel_scores = self.concat_torch(rel_scores, tmp_rel_score)
                            rel_labels.append(coarse2ids[rel])
                    # 2. tran: scores and labels
                    tmp_tran_score = self.model.score_tran(session_gold)
                    tran_labels.append(action2ids[tran])
                    tran_scores = self.concat_torch(tran_scores, tmp_tran_score)
                    session_gold, angle_prop_all = self.model(session_gold, transition_rel)
                loss_tran += loss_func_tran(tran_scores, torch.Tensor(tran_labels).long())
                if TRAIN_NR:
                    loss_nr = loss_nr + constraint_loss(nr_scores, torch.Tensor(nr_labels).long())
                    if TRAIN_NUCL_CONSTRAINT:
                        loss_nucl += loss_func_nucl(nucl_scores, torch.Tensor(nucl_labels).long())
                else:
                    loss_nucl += loss_func_nucl(nucl_scores, torch.Tensor(nucl_labels).long())
                    loss_rel = loss_rel + loss_func_rel(rel_scores, torch.Tensor(rel_labels).long())
                # batch learn
                if iter_count % BATCH_SIZE_spinn == 0 and iter_count > 0:
                    p.update(pro_idx)
                    pro_idx += 1
                    batch_count += 1
                    optimizer.zero_grad()
                    loss_tran.backward(retain_graph=True)
                    optimizer.step()
                    if TRAIN_NR:
                        if TRAIN_NUCL_CONSTRAINT:
                            loss_nr = loss_nr + CONSTRAINT_LAMBDA * loss_nucl  # 最大化 nuclearity的边际效应
                        optimizer.zero_grad()
                        loss_nr.backward()
                        optimizer.step()
                    else:
                        optimizer.zero_grad()
                        loss_nucl.backward(retain_graph=True)
                        optimizer.step()
                        optimizer.zero_grad()
                        loss_rel.backward()
                        optimizer.step()
                    loss_tran, loss_nucl, loss_rel, loss_nr = 0., 0., 0., 0.
                    # 评测
                    if batch_count % self.skip_steps == 0:
                        p.finish()
                        if CROSS_VAL:
                            better = self.evaluate(trees_eval_path=self.dev_trees_path, type_="dev", save_per=True)
                            # 在得到更好的模型的情况下执行对测试集的预测，问题是选择三个模型还是随着学习选择最好的指标
                            # 初步确立方案2：dev选模型，每次选到在span or nucl or rel上更好的模型就用它去跑test并记录更好的分数
                            if better:
                                self.evaluate(trees_eval_path=self.test_trees_path, type_="test", save_per=False)
                        else:
                            self.evaluate(trees_eval_path=self.test_trees_path, type_="test", save_per=True)
                        self.report(epoch, iter_count, test_desc, group_name)
                        if batch_count > self.skip_boundary and self.skip_steps > SKIP_STEP_min:
                            self.skip_steps -= SKIP_REDUCE_UNIT
                            self.skip_boundary += SKIP_BOUNDARY
                        # 开启新的进度
                        p.start(self.skip_steps)
                        pro_idx = 1
        # 存储
        self.save_eval_data()

    def evaluate(self, trees_eval_path, type_="test", save_per=False):
        """ 评测
        """
        self.model.eval()
        better = False
        self.parser = Parser(self.model)
        trees_eval = load_data(trees_eval_path)
        trees_pred = self.parser.parsing_all(trees_eval)
        # del self.parser
        # gc.collect()  # 释放内存
        trees_eval_pred = []
        for tree_ in trees_pred:
            trees_eval_pred.append(tree_obj(tree_))
        if type_ == "test":
            self.metric.eval_(self.test_trees, trees_eval_pred, self.model, type_=type_, save_per=save_per)
        else:
            better = self.metric.eval_(self.dev_trees, trees_eval_pred, self.model, type_=type_, save_per=save_per)
        return better

    def report(self, epoch, iter_count, test_desc="", group_name=None):
        """ 对评测结果的实时监控
        """
        report_info = []
        group_name = "dev & test" if group_name is None else group_name
        report_info.append(" version: " + str(VERSION) + " --Tran_LOSS V" + str(TRAN_LOSS_VERSION) +
                           " --REL_LOSS V" + str(REL_LOSS_VERSION) + " --- set: " + SET_of_version)
        epoch_str = "epoch: " + str(epoch+1) + "  iter_count: " + str(iter_count) + " skip_steps: " + \
                    str(self.skip_steps) + "group_name: " + group_name
        report_info.append(epoch_str)
        report_info += self.metric.get_scores()
        report_info.append(DESC + test_desc)
        report_info.append(OVER)
        self.output_report(report_info=report_info)

    def output_report(self, report_info=None):
        """ 打印中途预测信息
        """
        for info in report_info:
            if self.count_write % 50 == 0:
                self.count_write += 1
                print_(info, self.log_file, write_=True)
            else:
                print_(info, self.log_file, write_=False)
        self.count_write += 1

    def save_eval_data(self):
        """ 将训练过程中在开发集上面的三个指标随训练次数的准确率分布情况进行存储，供后期数据分析
        """
        save_data((self.metric.span_perf_long, self.metric.nucl_perf_long, self.metric.rel_perf_long), DEV_DISTRIBUTE)
        print_("the precision of both dev & test have been saved.", self.log_file)

    @staticmethod
    def restore(folder):
        """ Desc: Load the model.
        """
        model = load_data(os.path.join(folder, "model.pickle"))
        model.model = load_data(os.path.join(folder, "torch.bin"))
        return model

    @staticmethod
    def oracle(tree):
        """ Desc: Back_traverse a gold tree of rst-dt
            Input:  The tree object of rst_tree
            Output: Temp transition.
        """
        for node in tree.nodes:
            if node.left_child is not None and node.right_child is not None:
                yield (REDUCE + "-" + node.child_NS_rel, node.child_rel)
            else:
                yield SHIFT
