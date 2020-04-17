# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date: 2018/4/3
@Description:  parser_model
"""
import numpy as np
import torch
from utils.file_util import *
from config import MODELS2SAVE, VERSION, SET_of_version, REL_coarse2ids, CROSS_VAL, SAVE_MODEL


class Performance(object):
    def __init__(self, percision, recall):
        self.percision = percision
        self.recall = recall


class Metrics_Origin(object):
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.span_perf = []
        self.nuc_perf = []
        self.rela_perf = []
        # 存储总数
        self.span_true_all = 0.
        self.nuc_true_all = 0.
        self.rel_true_all = 0.
        self.span_all = 0.
        self.nuc_all = 0.
        self.rel_all = 0.

        # 最大值存储
        self.span_dev_max_mi = 0
        self.nucleas_dev_max_mi = 0
        self.relation_dev_max_mi = 0
        self.span_dev_max_ma = 0
        self.nucleas_dev_max_ma = 0
        self.relation_dev_max_ma = 0

        self.span_test_max_mi = 0
        self.nucleas_test_max_mi = 0
        self.relation_test_max_mi = 0
        self.span_test_max_ma = 0
        self.nucleas_test_max_ma = 0
        self.relation_test_max_ma = 0

        # 记录dev岁训练轮数的三个指标的分布情况
        self.span_pre_dev_mi = []
        self.nucl_pre_dev_mi = []
        self.rel_pre_dev_mi = []
        self.span_pre_dev_ma = []
        self.nucl_pre_dev_ma = []
        self.rel_pre_dev_ma = []

        # 记录 在测试集上准确率随开发机准确率的变化分析
        self.span_pre_test_mi = []
        self.nucl_pre_test_mi = []
        self.rel_pre_test_mi = []
        self.span_pre_test_ma = []
        self.nucl_pre_test_ma = []
        self.rel_pre_test_ma = []

        # 是否存储的标志
        self.remove_flag_mi = None
        self.remove_flag_ma = None

    def init_all(self):
        """
        忘记初始化了！！！！！！
        :return:
        """
        self.span_true_all = 0.
        self.nuc_true_all = 0.
        self.rel_true_all = 0.
        self.span_all = 0.
        self.nuc_all = 0.
        self.rel_all = 0.

    def eval_(self, goldtrees, predtrees, model, type_="dev"):
        """
        评测主函数
        :param type_: 默认开发集
        :param goldtrees:
        :param predtrees:
        :param model:
        :return:
        """
        self.span_perf = []
        self.nuc_perf = []
        self.rela_perf = []
        self.init_all()  # 必须初始化
        for idx in range(len(goldtrees)):
            tmp_goldspan_ids, _ = self.get_span(goldtrees[idx])
            tmp_predspan_ids, _ = self.get_span(predtrees[idx])
            tmp_goldspan_ns_ids, _ = self.get_span_ns(goldtrees[idx])
            tmp_predspan_ns_ids, _ = self.get_span_ns(predtrees[idx])
            tmp_goldspan_rel_ids = self.get_span_rel_ids(goldtrees[idx])
            tmp_predspan_rel_ids = self.get_span_rel_ids(predtrees[idx])
            self.eval_one(tmp_goldspan_ids, tmp_predspan_ids, tmp_goldspan_ns_ids, tmp_predspan_ns_ids,
                          tmp_goldspan_rel_ids, tmp_predspan_rel_ids)
        self.report(model=model, type_=type_)

    def eval_one(self, tmp_goldspan_ids, tmp_predspan_ids, tmp_goldspan_ns_ids, tmp_predspan_ns_ids,
                 tmp_goldspan_rel_ids, tmp_predspan_rel_ids):
        """
            compute the number of span in gold and pred for F and P.
        """
        # print("开始评测...")
        # span
        allspan = [span for span in tmp_goldspan_ids if span in tmp_predspan_ids]
        allspan_gold_idx = [tmp_goldspan_ids.index(span) for span in allspan]
        allspan_pred_idx = [tmp_predspan_ids.index(span) for span in allspan]
        # ns
        all_goldspan_ns = [tmp_goldspan_ns_ids[idx] for idx in allspan_gold_idx]
        all_predspan_ns = [tmp_predspan_ns_ids[idx] for idx in allspan_pred_idx]
        # rel
        all_goldspan_rel = [tmp_goldspan_rel_ids[idx] for idx in allspan_gold_idx]
        all_predspan_rel = [tmp_predspan_rel_ids[idx] for idx in allspan_pred_idx]

        # print("span评测中...")
        p_1, r_1 = 0.0, 0.0
        for span in allspan:
            if span in tmp_goldspan_ids:
                p_1 += 1.0
            if span in tmp_predspan_ids:
                r_1 += 1.0
        # micro
        self.span_true_all += (p_1 - 1)
        self.span_all += (len(tmp_goldspan_ids) - 1)
        # macro
        p = (p_1 - 1) / (len(tmp_goldspan_ids) - 1)
        self.span_perf.append(p)

        # "nuclearity评测中..."
        allspan_ns_count = sum(np.equal(all_goldspan_ns, all_predspan_ns))  # 看在相同的span上面标注的NS有多少是与标准一致
        # micro
        self.nuc_true_all += (allspan_ns_count - 1)
        self.nuc_all += (len(tmp_goldspan_ids) - 1)
        # macro
        p = (allspan_ns_count - 1) / (len(tmp_goldspan_ids) - 1)
        self.nuc_perf.append(p)

        # print("relation评测中...")
        if all_goldspan_rel is not None:
            allspan_rel_count = sum(np.equal(all_goldspan_rel, all_predspan_rel))
            # micro
            self.rel_true_all += (allspan_rel_count - 1)
            self.rel_all += (len(tmp_goldspan_ids) - 1)
            # macro
            p = (allspan_rel_count - 1) / (len(tmp_goldspan_ids) - 1)
            self.rela_perf.append(p)

    def report(self, model, type_="dev"):
        """
            汇总计算
        """
        p_span_micro = self.span_true_all / self.span_all
        p_span_macro = np.array(self.span_perf).mean()
        p_ns_micro = self.nuc_true_all / self.nuc_all
        p_ns_macro = np.array(self.nuc_perf).mean()
        p_rel_micro = self.rel_true_all / self.rel_all
        p_rel_macro = np.array(self.rela_perf).mean()
        span_file_name = "/test_span_max_model.pth" if type_ == "test" else "/dev_span_max_model.pth"
        ns_file_name = "/test_nucl_max_model.pth" if type_ == "test" else "/dev_nucl_max_model.pth"
        rel_file_name = "/test_rel_max_model.pth" if type_ == "test" else "/dev_rel_max_model.pth"

        if type_ == "dev":
            self.span_pre_dev_mi.append(p_span_micro)
            self.span_pre_dev_ma.append(p_span_macro)
            self.nucl_pre_dev_mi.append(p_ns_micro)
            self.nucl_pre_dev_ma.append(p_ns_macro)
            self.rel_pre_dev_mi.append(p_rel_micro)
            self.rel_pre_dev_ma.append(p_rel_macro)
        else:
            self.span_pre_test_mi.append(p_span_micro)
            self.span_pre_test_ma.append(p_span_macro)
            self.nucl_pre_test_mi.append(p_ns_micro)
            self.nucl_pre_test_ma.append(p_ns_macro)
            self.rel_pre_test_mi.append(p_rel_micro)
            self.rel_pre_test_ma.append(p_rel_macro)

        # 针对两种指标的最大值更新程序
        tmp_span_max_mi = self.span_test_max_mi if type_ == "test" else self.span_dev_max_mi
        tmp_ns_max_mi = self.nucleas_test_max_mi if type_ == "test" else self.nucleas_dev_max_mi
        tmp_rel_max_mi = self.relation_test_max_mi if type_ == "test" else self.relation_dev_max_mi
        tmp_span_max_ma = self.span_test_max_ma if type_ == "test" else self.span_dev_max_ma
        tmp_ns_max_ma = self.nucleas_test_max_ma if type_ == "test" else self.nucleas_dev_max_ma
        tmp_rel_max_ma = self.relation_test_max_ma if type_ == "test" else self.relation_dev_max_ma

        # 确定是否存储数据
        # self.remove_flag_mi = False if ((p_span_micro > tmp_span_max_mi) or (p_ns_micro > tmp_ns_max_mi) or
        #                                 (p_rel_micro > tmp_rel_max_mi)) else True
        # self.remove_flag_ma = False if ((p_span_macro > tmp_span_max_ma) or (p_ns_macro > tmp_ns_max_ma) or
        #                                 (p_rel_macro > tmp_rel_max_ma)) else True
        # 下面对最大值更新
        # 1. span
        if p_span_micro > tmp_span_max_mi:
            if type_ == "test":
                self.span_test_max_mi = p_span_micro
            else:
                # dev
                self.span_dev_max_mi = p_span_micro
            if SAVE_MODEL:
                self.save_model(file_name=span_file_name, model=model)
        # span_str = 'F1 score span(micro): ' + str(p_span_micro)
        # print_(span_str, self.log_file)

        if p_span_macro > tmp_span_max_ma:
            if type_ == "test":
                self.span_test_max_ma = p_span_macro
            else:
                # dev
                self.span_dev_max_ma = p_span_macro
            if SAVE_MODEL:
                self.save_model(file_name=span_file_name, model=model)
        # span_str = 'F1 score span(macro): ' + str(p_span_macro)
        # print_(span_str, self.log_file)

        # 2. nuclearity
        if p_ns_micro > tmp_ns_max_mi:
            if type_ == "test":
                self.nucleas_test_max_mi = p_ns_micro
            else:
                self.nucleas_dev_max_mi = p_ns_micro
            if SAVE_MODEL:
                self.save_model(file_name=ns_file_name, model=model)
        # ns_str = 'F1 score nuclearity(micro): ' + str(p_ns_micro)
        # print_(ns_str, self.log_file)

        if p_ns_macro > tmp_ns_max_ma:
            if type_ == "test":
                self.nucleas_test_max_ma = p_ns_macro
            else:
                self.nucleas_dev_max_ma = p_ns_macro
            if SAVE_MODEL:
                self.save_model(file_name=ns_file_name, model=model)
        # ns_str = 'F1 score nuclearity(macro): ' + str(p_ns_macro)
        # print_(ns_str, self.log_file)

        # 3. relation
        if p_rel_micro > tmp_rel_max_mi:
            if type_ == "test":
                self.relation_test_max_mi = p_rel_micro
            else:
                self.relation_dev_max_mi = p_rel_micro
            if SAVE_MODEL:
                self.save_model(file_name=rel_file_name, model=model)
        # rel_str = 'F1 score relation(micro): ' + str(p_rel_micro)
        # print_(rel_str, self.log_file)

        if p_rel_macro > tmp_rel_max_ma:
            if type_ == "test":
                self.relation_test_max_ma = p_rel_macro
            else:
                self.relation_dev_max_ma = p_rel_macro
            if SAVE_MODEL:
                self.save_model(file_name=rel_file_name, model=model)
        # rel_str = 'F1 score relation(macro): ' + str(p_rel_macro)
        # print_(rel_str, self.log_file)

    def remove_saved_mi(self):
        """
        对不需要存储的数据直接pop出去
        :return:
        """
        if CROSS_VAL:
            self.span_pre_dev_mi.pop()
            self.nucl_pre_dev_mi.pop()
            self.rel_pre_dev_mi.pop()
        self.span_pre_test_mi.pop()
        self.nucl_pre_test_mi.pop()
        self.rel_pre_test_mi.pop()
        self.remove_flag_mi = False  # 默认情况不剔除

    def remove_saved_ma(self):
        """
        对不需要存储的数据直接pop出去
        :return:
        """
        if CROSS_VAL:
            self.span_pre_dev_ma.pop()
            self.nucl_pre_dev_ma.pop()
            self.rel_pre_dev_ma.pop()
        self.span_pre_test_ma.pop()
        self.nucl_pre_test_ma.pop()
        self.rel_pre_test_ma.pop()
        self.remove_flag_ma = False  # 默认情况不剔除

    @staticmethod
    def save_model(file_name, model):
        # 存储
        dir2save = MODELS2SAVE + "/v" + str(VERSION) + "_set" + str(SET_of_version)
        safe_mkdir(dir2save)
        save_path = dir2save + file_name
        torch.save(model.state_dict(), save_path)

    @staticmethod
    def get_span(tree_):
        """
        获取每棵树的各自的span_ids
        :param tree_:
        :return:
        """
        count_edus = 0
        span_ids = []
        for node in tree_.nodes:
            if node.left_child is None and node.right_child is None:
                count_edus += 1
            span_ids.append(node.temp_edu_span)
        return span_ids, count_edus


    @staticmethod
    def get_span_ns(tree_):
        """
        获取每棵树的各自的span_ids
        :param tree_:
        :return:
        """
        ns_dict = {"Satellite": 0, "Nucleus": 1, "Root": 2}
        count_edus = 0
        span_ns_ids = []
        for node in tree_.nodes:
            if node.left_child is None and node.right_child is None:
                count_edus += 1
            span_ns_ids.append(ns_dict[node.type])
        return span_ns_ids, count_edus


    @staticmethod
    def get_span_rel_ids(tree_):
        """
        :param tree_:
        :return:
        """
        coarse2ids = load_data(REL_coarse2ids)
        span_rel_ids = []
        for node in tree_.nodes:
            span_rel_ids.append(coarse2ids[node.rel])
        return span_rel_ids

    def get_scores(self):
        """
        :return:
        """
        report_info = list()
        report_info.append("==================== TEST =====================")
        report_info.append("(Micro) S: " + str(self.span_test_max_mi) + " -- N: " + str(self.nucleas_test_max_mi) +
                           " -- R:" + str(self.relation_test_max_mi))
        report_info.append("(Macro) S: " + str(self.span_test_max_ma) + " -- N: " + str(self.nucleas_test_max_ma) +
                           " -- R:" + str(self.relation_test_max_ma))
        return report_info
