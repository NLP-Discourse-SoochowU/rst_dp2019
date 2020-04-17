# -*- coding: utf-8 -*-

"""
@Author: Lyzhang
@Date:
@Description: micro + standard parseval
"""
import numpy as np
import torch
from utils.file_util import *
from config import *


class Metrics(object):
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.true_all = [0., 0., 0., 0.]  # span nucl rel f
        self.span_all = 0.

        # dev数据集最大值存储, Inner_mi
        self.dev_span_max, self.dev_nucl_max, self.dev_rel_max, self.dev_f_max = 0, 0, 0, 0

        # test数据集最大值存储, Inner_mi
        self.test_span_max, self.test_nucl_max, self.test_rel_max, self.test_f_max = 0, 0, 0, 0  # Inner_mi

        # 提供准确率的存储，提供拟合的数据分析工作, Inner_mi
        self.span_perf_long, self.nucl_perf_long, self.rel_perf_long = [], [], []

        # 存储最佳模型得分的各个指标对应的剩余分数
        self.test_s_m_scores = [0., 0., 0., 0.]  # s, n, r, f
        self.test_n_m_scores = [0., 0., 0., 0.]
        self.test_r_m_scores = [0., 0., 0., 0.]
        self.test_f_m_scores = [0., 0., 0., 0.]
        # dev
        self.dev_s_m_scores = [0., 0., 0., 0.]  # s, n, r, f
        self.dev_n_m_scores = [0., 0., 0., 0.]
        self.dev_r_m_scores = [0., 0., 0., 0.]
        self.dev_f_m_scores = [0., 0., 0., 0.]

    def init_all(self):
        """
        初始化
        :return:
        """
        self.true_all, self.span_all = [0., 0., 0., 0.], 0.

    def eval_(self, goldtrees, predtrees, model, type_="dev", save_per=False):
        """
        评测主函数，针对dev集合如果找到更好的模型，返回True作为test集合学习的继续
        :return:
        """
        self.init_all()  # 初始化
        for idx in range(len(goldtrees)):
            goldspan_ids, goldspan_ns_ids, goldspan_rel_ids = self.get_all_span_info(goldtrees[idx])
            predspan_ids, predspan_ns_ids, predspan_rel_ids = self.get_all_span_info(predtrees[idx])
            # 对当前篇章评测计算
            self.eval_all(goldspan_ids, predspan_ids, goldspan_ns_ids, predspan_ns_ids, goldspan_rel_ids,
                          predspan_rel_ids)
        better = self.report(model=model, type_=type_, predtrees=predtrees, save_per=save_per)
        return better

    def eval_all(self, gold_s_ids, pred_s_ids, gold_ns_ids, pred_ns_ids, gold_rel_ids, pred_rel_ids):
        """
        compute the number of span in gold and pred for F and P.
        goldspan_ids: 包含所有span的ids
        inner_goldspan_ids:  各个内部节点中标签不为span的节点区域对应id
        :return:
        """
        # span
        # 获取包含叶节点的所有预测正确的span并对span打标签与标准和预测树中对应位置一一对应
        allspan = [span for span in gold_s_ids if span in pred_s_ids]
        allspan_gold_idx = [gold_s_ids.index(span) for span in allspan]
        allspan_pred_idx = [pred_s_ids.index(span) for span in allspan]
        # ns
        all_gold_ns = [gold_ns_ids[idx] for idx in allspan_gold_idx]
        all_pred_ns = [pred_ns_ids[idx] for idx in allspan_pred_idx]

        # rel
        all_gold_rel = [gold_rel_ids[idx] for idx in allspan_gold_idx]
        all_pred_rel = [pred_rel_ids[idx] for idx in allspan_pred_idx]

        # macro & micro
        span_len = float(len(gold_s_ids))
        self.compute_macro_micro_parseval(allspan, all_gold_ns, all_pred_ns, all_gold_rel, all_pred_rel, span_len)

    def compute_macro_micro_parseval(self, allspan, all_gold_ns, all_pred_ns, all_gold_rel, all_pred_rel, span_len):
        """
        standard parseval
        :return:
        """
        ns_equal = np.equal(all_gold_ns, all_pred_ns)
        rel_equal = np.equal(all_gold_rel, all_pred_rel)
        f_equal = [ns_equal[idx] and rel_equal[idx] for idx in range(len(ns_equal))]
        # standard parseval 里面均不必去除root对应的span
        s_pred, ns_pred, rel_pred, f_pred = len(allspan), sum(ns_equal), sum(rel_equal), sum(f_equal)
        # micro
        self.true_all[0] += s_pred
        self.true_all[1] += ns_pred
        self.true_all[2] += rel_pred
        self.true_all[3] += f_pred
        # span和其他预测分别累加span个数（每棵树的span预测不包含root）
        self.span_all += span_len

    @staticmethod
    def get_all_span_info(tree_):
        """
        获取每棵树的各自的 span_ids，同时给出不包含 SPAN关系的 SPAN 列表 (分别供 span 和 (nucl, rel) 的预测 )
        修正：对于关系等标签写在父节点上面的情况，获取span的时候只需要获取内部节点和根节点即可，不用纠结span标签
        :param tree_: 对当前一个tree获取所有信息
        :return:
        """
        # rel2ids = load_data(REL_coarse2ids)
        span_ids = []
        span_ns_ids = []
        span_rel_ids = []
        for node in tree_.nodes:
            if node.left_child is not None and node.right_child is not None:
                span_ids.append(node.temp_edu_span)  # 获取内部节点中为核心的节点的区域或者root涵盖的区域
                span_ns_ids.append(nucl2ids[node.child_NS_rel])  # 将所有内部节点的所有ns的ids进行搜集
                span_rel_ids.append(coarse2ids[node.child_rel])  # 搜集rel不是"span"的所有节点对应的rel的ids
        return span_ids, span_ns_ids, span_rel_ids

    def report(self, model, type_="dev", predtrees=None, save_per=False):
        """
            汇总
        """
        report_info = []
        # micro
        p_span, p_ns, p_rel, p_f = (self.true_all[idx] / self.span_all for idx in range(4))
        # 获取历史最大值：span_max: (mi inner_mi ma inner_ma)
        span_max, nucl_max, rel_max, f_max = self.get_all_max(type_=type_)

        # 整合
        if save_per:
            self.update_per_long(p_span, p_ns, p_rel)
        # 最大值更新
        better = self.update_all_max(p_span, p_ns, p_rel, p_f, span_max, nucl_max, rel_max, f_max, report_info, type_)
        # 模型存储
        self.save_best_models(p_span, p_ns, p_rel, p_f, span_max, nucl_max, rel_max, f_max, model, type_, predtrees)
        # self.output_report(report_info)
        # 判定开发集是否得到更好的模型
        return better

    def get_all_max(self, type_="dev"):
        """
        获取往期最大值
        """
        # 针对两种指标的最大值更新程序
        span_max = self.test_span_max if type_ == "test" else self.dev_span_max
        nucl_max = self.test_nucl_max if type_ == "test" else self.dev_nucl_max
        rel_max = self.test_rel_max if type_ == "test" else self.dev_rel_max
        f_max = self.test_f_max if type_ == "test" else self.dev_f_max
        return span_max, nucl_max, rel_max, f_max

    def update_all_max(self, span_pre, nucl_pre, rel_pre, f_pre, span_max, nucl_max, rel_max, f_max, rep_info, type_):
        """
        根据xxx_pre和xxx_max对比更新最大值存储
        span_pre: (mi inner_mi ma inner_ma)
        :return:
        """
        better = False
        # span
        if span_pre > span_max:
            if type_ == "test":
                self.test_span_max = span_pre
                self.test_s_m_scores = [span_pre, nucl_pre, rel_pre, f_pre]
            else:
                better = True
                self.dev_span_max = span_pre
                self.dev_s_m_scores = [span_pre, nucl_pre, rel_pre, f_pre]
        rep_info.append('span(inner_mi): ' + str(span_pre))
        # nucl
        if nucl_pre > nucl_max:
            if type_ == "test":
                self.test_nucl_max = nucl_pre
                self.test_n_m_scores = [span_pre, nucl_pre, rel_pre, f_pre]
            else:
                better = True
                self.dev_nucl_max = nucl_pre
                self.dev_n_m_scores = [span_pre, nucl_pre, rel_pre, f_pre]
        rep_info.append('nucl(inner_mi): ' + str(nucl_pre))
        # rel
        if rel_pre > rel_max:
            if type_ == "test":
                self.test_rel_max = rel_pre
                self.test_r_m_scores = [span_pre, nucl_pre, rel_pre, f_pre]
            else:
                better = True
                self.dev_rel_max = rel_pre
                self.dev_r_m_scores = [span_pre, nucl_pre, rel_pre, f_pre]
        rep_info.append('rel(mi inner_mi ma inner_ma): ' + str(rel_pre))
        # F
        if f_pre > f_max:
            if type_ == "test":
                self.test_f_max = f_pre
                self.test_f_m_scores = [span_pre, nucl_pre, rel_pre, f_pre]
            else:
                better = True
                self.dev_f_max = f_pre
                self.dev_f_m_scores = [span_pre, nucl_pre, rel_pre, f_pre]
        rep_info.append('F(mi inner_mi ma inner_ma): ' + str(f_pre))
        return better

    def update_per_long(self, span_pre, nucl_pre, rel_pre):
        """
        提供曲线学习分析
        :param span_pre:
        :param nucl_pre:
        :param rel_pre:
        :return:
        """
        self.span_perf_long.append(span_pre)
        self.nucl_perf_long.append(nucl_pre)
        self.rel_perf_long.append(rel_pre)

    def save_best_models(self, span_pre, nucl_pre, rel_pre, f_pre, span_max, nucl_max, rel_max, f_max, model, type_,
                         predtrees):
        """
        待完善，最后决定只对micro_inner选到最好的模型进行存储
        :return:
        """
        if SAVE_MODEL:
            span_file_name = "/test_span_max_model.pth" if type_ == "test" else "/dev_span_max_model.pth"
            span_best_trees_parsed = "/test_span_trees.pkl" if type_ == "test" else "/dev_span_trees.pkl"
            ns_file_name = "/test_nucl_max_model.pth" if type_ == "test" else "/dev_nucl_max_model.pth"
            ns_best_trees_parsed = "/test_ns_trees.pkl" if type_ == "test" else "/dev_ns_trees.pkl"
            rel_file_name = "/test_rel_max_model.pth" if type_ == "test" else "/dev_rel_max_model.pth"
            rel_best_trees_parsed = "/test_rel_trees.pkl" if type_ == "test" else "/dev_rel_trees.pkl"
            f_file_name = "/test_f_max_model.pth" if type_ == "test" else "/dev_f_max_model.pth"
            f_best_trees_parsed = "/test_f_trees.pkl" if type_ == "test" else "/dev_f_trees.pkl"
            if span_pre > span_max:
                self.save_model(file_name=span_file_name, model=model)
                self.save_trees(file_name=span_best_trees_parsed, trees=predtrees)
            if nucl_pre > nucl_max:
                self.save_model(file_name=ns_file_name, model=model)
                self.save_trees(file_name=ns_best_trees_parsed, trees=predtrees)
            if rel_pre > rel_max:
                self.save_model(file_name=rel_file_name, model=model)
                self.save_trees(file_name=rel_best_trees_parsed, trees=predtrees)
            if f_pre > f_max:
                self.save_model(file_name=f_file_name, model=model)
                self.save_trees(file_name=f_best_trees_parsed, trees=predtrees)

    @staticmethod
    def save_model(file_name, model):
        dir2save = MODELS2SAVE + "/v" + str(VERSION) + "_set" + str(SET_of_version)
        safe_mkdir(MODELS2SAVE)
        safe_mkdir(dir2save)
        save_path = dir2save + file_name
        torch.save(model, save_path)

    @staticmethod
    def save_trees(file_name, trees):
        dir2save = MODELS2SAVE + "/v" + str(VERSION) + "_set" + str(SET_of_version)
        safe_mkdir(MODELS2SAVE)
        safe_mkdir(dir2save)
        save_path = dir2save + file_name
        save_data(trees, save_path)

    def output_report(self, report_info=None):
        """
        打印中途预测信息
        :return:
        """
        for info in report_info:
            print_(info, self.log_file)

    def write_bracket_all(self, allspan, all_goldspan_ns, all_goldspan_rel, all_predspan_ns, all_predspan_rel,
                          allspan_inner, all_inner_goldspan_ns, all_inner_goldspan_rel, all_inner_predspan_ns,
                          all_inner_predspan_rel, file_name):
        self.write_bracket(allspan, all_goldspan_ns, all_goldspan_rel, BRACKET_PATH, file_name=file_name)
        self.write_bracket(allspan, all_predspan_ns, all_predspan_rel, BRACKET_PATH + ".pre", file_name=file_name)
        self.write_bracket(allspan_inner, all_inner_goldspan_ns, all_inner_goldspan_rel, BRACKET_INNER_PATH,
                           file_name=file_name, type_="inner")
        self.write_bracket(allspan_inner, all_inner_predspan_ns, all_inner_predspan_rel, BRACKET_INNER_PATH + ".pre",
                           file_name=file_name, type_="inner")
        input("write ok")

    @staticmethod
    def write_bracket(span, ns, rel, file_path, file_name=None, type_="all"):
        lines = [file_name]
        for idx in range(len(span)):
            nucl = ns_dict_[ns[idx]] if type_ == "all" else ids2nucl[ns[idx]]
            tmp_line = "((" + str(span[idx][0]) + ", " + str(span[idx][1]) + "), " + nucl + ", " + \
                       ids2coarse[rel[idx]] + ")"
            lines.append(tmp_line)
        write_iterate(lines, file_path)

    def get_scores(self):
        """
        (mi, inner_mi, ma, inner_ma)
        :return:
        """
        report_info = []
        if CROSS_VAL:
            report_info.append("==================== DEV =====================")
            report_info.append("(All_MAX) S: " + str(self.dev_span_max) + " -- N: " + str(self.dev_nucl_max)
                               + " -- R:" + str(self.dev_rel_max) + " -- F:" + str(self.dev_f_max))
            report_info.append("(S_MAX) [" + str(self.dev_s_m_scores[0]) + ", " + str(self.dev_s_m_scores[1])
                               + ", " + str(self.dev_s_m_scores[2]) + ", " + str(self.dev_s_m_scores[3]) + "]")
            report_info.append("(N_MAX) [" + str(self.dev_n_m_scores[0]) + ", " + str(self.dev_n_m_scores[1])
                               + ", " + str(self.dev_n_m_scores[2]) + ", " + str(self.dev_n_m_scores[3]) + "]")
            report_info.append("(R_MAX) [" + str(self.dev_r_m_scores[0]) + ", " + str(self.dev_r_m_scores[1])
                               + ", " + str(self.dev_r_m_scores[2]) + ", " + str(self.dev_r_m_scores[3]) + "]")
            report_info.append("(F_MAX) [" + str(self.dev_f_m_scores[0]) + ", " + str(self.dev_f_m_scores[1])
                               + ", " + str(self.dev_f_m_scores[2]) + ", " + str(self.dev_f_m_scores[3]) + "]")
        report_info.append("==================== TEST =====================")
        report_info.append("(ALL_MAX) S: " + str(self.test_span_max) + " -- N: " + str(self.test_nucl_max)
                           + " -- R:" + str(self.test_rel_max) + " -- F:" + str(self.test_f_max))
        report_info.append("(S_MAX) [" + str(self.test_s_m_scores[0]) + ", " + str(self.test_s_m_scores[1])
                           + ", " + str(self.test_s_m_scores[2]) + ", " + str(self.test_s_m_scores[3]) + "]")
        report_info.append("(N_MAX) [" + str(self.test_n_m_scores[0]) + ", " + str(self.test_n_m_scores[1])
                           + ", " + str(self.test_n_m_scores[2]) + ", " + str(self.test_n_m_scores[3]) + "]")
        report_info.append("(R_MAX) [" + str(self.test_r_m_scores[0]) + ", " + str(self.test_r_m_scores[1])
                           + ", " + str(self.test_r_m_scores[2]) + ", " + str(self.test_r_m_scores[3]) + "]")
        report_info.append("(F_MAX) [" + str(self.test_f_m_scores[0]) + ", " + str(self.test_f_m_scores[1])
                           + ", " + str(self.test_f_m_scores[2]) + ", " + str(self.test_f_m_scores[3]) + "]")
        return report_info
