# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date: 2018/4/3
@Description:  parser_model
"""
import numpy as np
import torch
from utils.file_util import *
from config import *


class Metrics(object):
    def __init__(self, log_file=None):
        self.log_file = log_file
        self.span_perf = []
        self.nuc_perf = []
        self.rel_perf = []
        self.f_perf = []
        self.span_perf_inner = []
        self.nuc_perf_inner = []
        self.rel_perf_inner = []
        self.f_perf_inner = []
        self.true_all = [0., 0., 0., 0.]  # span nucl rel f
        self.true_all_inner = [0., 0., 0., 0.]  # span nucl rel f
        self.span_all = 0.
        self.span_all_inner = 0.

        self.dev_span_max = [0, 0, 0, 0]  # mi Inner_mi ma Inner_ma
        self.dev_nucl_max = [0, 0, 0, 0]
        self.dev_rel_max = [0, 0, 0, 0]
        self.dev_f_max = [0, 0, 0, 0]

        self.test_span_max = [0, 0, 0, 0]  # mi Inner_mi ma Inner_ma
        self.test_nucl_max = [0, 0, 0, 0]
        self.test_rel_max = [0, 0, 0, 0]
        self.test_f_max = [0, 0, 0, 0]

        self.span_perf_long = [[], [], [], []]  # mi Inner_mi ma Inner_ma
        self.nucl_perf_long = [[], [], [], []]
        self.rel_perf_long = [[], [], [], []]

    def init_all(self):
        # micro
        self.true_all = [0., 0., 0., 0.]
        self.span_all = 0.
        self.true_all_inner = [0., 0., 0., 0.]
        self.span_all_inner = 0.
        # macro
        self.span_perf = []
        self.nuc_perf = []
        self.rel_perf = []
        self.f_perf = []
        self.span_perf_inner = []
        self.nuc_perf_inner = []
        self.rel_perf_inner = []
        self.f_perf_inner = []

    def eval_(self, goldtrees, predtrees, model, type_="dev", save_per=False):
        self.init_all()
        for idx in range(len(goldtrees)):
            goldspan_ids, goldspan_ns_ids, goldspan_rel_ids, inner_goldspan_ids, inner_goldspan_ns_ids, \
                inner_goldspan_rel_ids = self.get_all_span_info(goldtrees[idx])
            predspan_ids, predspan_ns_ids, predspan_rel_ids, inner_predspan_ids, inner_predspan_ns_ids, \
                inner_predspan_rel_ids = self.get_all_span_info(predtrees[idx])
            self.eval_all(goldspan_ids, predspan_ids, goldspan_ns_ids, predspan_ns_ids, goldspan_rel_ids,
                          predspan_rel_ids, inner_goldspan_ids, inner_predspan_ids, inner_goldspan_ns_ids,
                          inner_predspan_ns_ids, inner_goldspan_rel_ids, inner_predspan_rel_ids,
                          file_name=goldtrees[idx].file_name)
        better = self.report(model=model, type_=type_, predtrees=predtrees, save_per=save_per)
        return better

    def eval_all(self, goldspan_ids, predspan_ids, goldspan_ns_ids, predspan_ns_ids, goldspan_rel_ids, predspan_rel_ids,
                 inner_goldspan_ids, inner_predspan_ids, inner_goldspan_ns_ids, inner_predspan_ns_ids,
                 inner_goldspan_rel_ids, inner_predspan_rel_ids, file_name=None):
        # span
        allspan = [span for span in goldspan_ids if span in predspan_ids]
        allspan_gold_idx = [goldspan_ids.index(span) for span in allspan]
        allspan_pred_idx = [predspan_ids.index(span) for span in allspan]
        allspan_inner = [span for span in inner_goldspan_ids if span in inner_predspan_ids]
        allspan_inner_gold_idx = [inner_goldspan_ids.index(span) for span in allspan_inner]
        allspan_inner_pred_idx = [inner_predspan_ids.index(span) for span in allspan_inner]

        # ns
        all_goldspan_ns = [goldspan_ns_ids[idx] for idx in allspan_gold_idx]
        all_predspan_ns = [predspan_ns_ids[idx] for idx in allspan_pred_idx]
        all_inner_goldspan_ns = [inner_goldspan_ns_ids[idx] for idx in allspan_inner_gold_idx]
        all_inner_predspan_ns = [inner_predspan_ns_ids[idx] for idx in allspan_inner_pred_idx]

        # rel
        all_goldspan_rel = [goldspan_rel_ids[idx] for idx in allspan_gold_idx]
        all_predspan_rel = [predspan_rel_ids[idx] for idx in allspan_pred_idx]
        all_inner_goldspan_rel = [inner_goldspan_rel_ids[idx] for idx in allspan_inner_gold_idx]
        all_inner_predspan_rel = [inner_predspan_rel_ids[idx] for idx in allspan_inner_pred_idx]

        # self.write_bracket_all(allspan, all_goldspan_ns, all_goldspan_rel, all_predspan_ns, all_predspan_rel,
        #                        allspan_inner, all_inner_goldspan_ns, all_inner_goldspan_rel, all_inner_predspan_ns,
        #                        all_inner_predspan_rel, file_name)

        # macro & micro
        true_len, span_len = float(len(allspan)), float(len(goldspan_ids))
        self.compute_macro_micro_original(true_len, span_len, all_goldspan_ns, all_predspan_ns, all_goldspan_rel,
                                          all_predspan_rel)
        true_len, span_len = float(len(allspan_inner)), float(len(inner_goldspan_ids))
        self.compute_macro_micro_parseval(true_len, span_len, all_inner_goldspan_ns, all_inner_predspan_ns,
                                          all_inner_goldspan_rel, all_inner_predspan_rel)

    def compute_macro_micro_original(self, true_len, span_len, all_goldspan_ns, all_predspan_ns, all_goldspan_rel,
                                     all_predspan_rel):
        ns_equal = np.equal(all_goldspan_ns, all_predspan_ns)
        rel_equal = np.equal(all_goldspan_rel, all_predspan_rel)
        f_equal = [ns_equal[idx] and rel_equal[idx] for idx in range(len(ns_equal))]
        ns_equal_all = sum(ns_equal) - 1
        rel_equal_all = sum(rel_equal) - 1
        f_equal_all = sum(f_equal) - 1
        true_len -= 1
        span_len -= 1
        p_span = true_len / span_len
        p_ns = ns_equal_all / span_len
        p_rel = rel_equal_all / span_len
        p_f = f_equal_all / span_len
        # macro
        self.span_perf.append(p_span)
        self.nuc_perf.append(p_ns)
        self.rel_perf.append(p_rel)
        self.f_perf.append(p_f)
        #  micro
        self.true_all[0] += true_len
        self.true_all[1] += ns_equal_all
        self.true_all[2] += rel_equal_all
        self.true_all[3] += f_equal_all
        self.span_all += span_len

    def compute_macro_micro_parseval(self, true_len, span_len, all_goldspan_ns, all_predspan_ns, all_goldspan_rel,
                                     all_predspan_rel):
        ns_equal = np.equal(all_goldspan_ns, all_predspan_ns)
        rel_equal = np.equal(all_goldspan_rel, all_predspan_rel)
        f_equal = [ns_equal[idx] and rel_equal[idx] for idx in range(len(ns_equal))]
        ns_equal_all = sum(ns_equal)
        rel_equal_all = sum(rel_equal)
        f_equal_all = sum(f_equal)
        p_span = true_len / span_len
        p_ns = ns_equal_all / span_len
        p_rel = rel_equal_all / span_len
        p_f = f_equal_all / span_len
        # macro
        self.span_perf_inner.append(p_span)
        self.nuc_perf_inner.append(p_ns)
        self.rel_perf_inner.append(p_rel)
        self.f_perf_inner.append(p_f)
        # micro
        self.true_all_inner[0] += true_len
        self.true_all_inner[1] += ns_equal_all
        self.true_all_inner[2] += rel_equal_all
        self.true_all_inner[3] += f_equal_all
        self.span_all_inner += span_len

    @staticmethod
    def get_all_span_info(tree_):
        # rel2ids = load_data(REL_coarse2ids)
        span_ids = []
        span_ns_ids = []
        span_rel_ids = []
        inner_span_ids = []
        inner_span_ns_ids = []
        inner_r_rel_ids = []
        for node in tree_.nodes:
            if node.left_child is not None and node.right_child is not None:
                inner_span_ids.append(node.temp_edu_span)
                inner_span_ns_ids.append(nucl2ids[node.child_NS_rel])
                inner_r_rel_ids.append(coarse2ids[node.child_rel])
            span_ids.append(node.temp_edu_span)
            span_ns_ids.append(ns_dict[node.type])
            span_rel_ids.append(coarse2ids[node.rel])
        return span_ids, span_ns_ids, span_rel_ids, inner_span_ids, inner_span_ns_ids, inner_r_rel_ids

    def get_all_max(self, type_="dev"):
        """ span_max, nucl_max, rel_max, f_max
            span_max: (mi inner_mi ma inner_ma)
        """
        span_max = [self.test_span_max[idx] for idx in range(4)] if type_ == "test" else \
            [self.dev_span_max[idx] for idx in range(4)]

        nucl_max = [self.test_nucl_max[idx] for idx in range(4)] if type_ == "test" else \
            [self.dev_nucl_max[idx] for idx in range(4)]

        rel_max = [self.test_rel_max[idx] for idx in range(4)] if type_ == "test" else \
            [self.dev_rel_max[idx] for idx in range(4)]

        f_max = [self.test_f_max[idx] for idx in range(4)] if type_ == "test" else \
            [self.dev_f_max[idx] for idx in range(4)]
        return span_max, nucl_max, rel_max, f_max

    def update_all_max(self, span_pre, nucl_pre, rel_pre, f_pre, span_max, nucl_max, rel_max, f_max, rep_info, type_):
        """ span_pre: (mi inner_mi ma inner_ma)
        """
        better = False
        for idx in range(4):
            # span
            if span_pre[idx] > span_max[idx]:
                if type_ == "test":
                    self.test_span_max[idx] = span_pre[idx]
                else:
                    better = True
                    self.dev_span_max[idx] = span_pre[idx]
            rep_info.append('span(mi inner_mi ma inner_ma): ' + str(span_pre[idx]))
            # nucl
            if nucl_pre[idx] > nucl_max[idx]:
                if type_ == "test":
                    self.test_nucl_max[idx] = nucl_pre[idx]
                else:
                    better = True
                    self.dev_nucl_max[idx] = nucl_pre[idx]
            rep_info.append('nucl(mi inner_mi ma inner_ma): ' + str(nucl_pre[idx]))
            # rel
            if rel_pre[idx] > rel_max[idx]:
                if type_ == "test":
                    self.test_rel_max[idx] = rel_pre[idx]
                else:
                    better = True
                    self.dev_rel_max[idx] = rel_pre[idx]
            rep_info.append('rel(mi inner_mi ma inner_ma): ' + str(rel_pre[idx]))
            # F
            if f_pre[idx] > f_max[idx]:
                if type_ == "test":
                    self.test_f_max[idx] = f_pre[idx]
                else:
                    better = True
                    self.dev_f_max[idx] = f_pre[idx]
            rep_info.append('F(mi inner_mi ma inner_ma): ' + str(f_pre[idx]))
        return better

    def report(self, model, type_="dev", predtrees=None, save_per=False):
        report_info = []
        # === All ===
        # macro
        p_span_macro, p_ns_macro, p_rel_macro, p_f_macro = np.array(self.span_perf).mean(), \
            np.array(self.nuc_perf).mean(), np.array(self.rel_perf).mean(), np.array(self.f_perf).mean()
        # micro
        p_span_micro, p_ns_micro, p_rel_micro, p_f_micro = (self.true_all[idx] / self.span_all for idx in range(4))
        # === Inner ===
        # macro
        p_span_r_macro, p_ns_r_macro, p_rel_r_macro, p_f_r_macro = np.array(self.span_perf_inner).mean(), \
            np.array(self.nuc_perf_inner).mean(), np.array(self.rel_perf_inner).mean(), np.array(self.f_perf_inner).mean()
        # micro
        p_span_r_micro = self.true_all_inner[0] / self.span_all_inner
        p_ns_r_micro, p_rel_r_micro, p_f_r_micro = (self.true_all_inner[idx] / self.span_all_inner for idx
                                                    in range(1, 4))
        # span_max: (mi inner_mi ma inner_ma)
        span_max, nucl_max, rel_max, f_max = self.get_all_max(type_=type_)

        # (mi inner_mi ma inner_ma)
        span_pre = [p_span_micro, p_span_r_micro, p_span_macro, p_span_r_macro]
        nucl_pre = [p_ns_micro, p_ns_r_micro, p_ns_macro, p_ns_r_macro]
        rel_pre = [p_rel_micro, p_rel_r_micro, p_rel_macro, p_rel_r_macro]
        f_pre = [p_f_micro, p_f_r_micro, p_f_macro, p_f_r_macro]
        if save_per:
            self.update_per_long(span_pre, nucl_pre, rel_pre)
        better = self.update_all_max(span_pre, nucl_pre, rel_pre, f_pre, span_max, nucl_max, rel_max, f_max, report_info
                                     , type_)
        self.save_best_models(span_pre, nucl_pre, rel_pre, f_pre, span_max, nucl_max, rel_max, f_max, model, type_,
                              predtrees)
        return better

    def update_per_long(self, span_pre, nucl_pre, rel_pre):
        for idx in range(4):
            self.span_perf_long[idx].append(span_pre[idx])
            self.nucl_perf_long[idx].append(nucl_pre[idx])
            self.rel_perf_long[idx].append(rel_pre[idx])

    def save_best_models(self, span_pre, nucl_pre, rel_pre, f_pre, span_max, nucl_max, rel_max, f_max, model, type_,
                         predtrees):
        if SAVE_MODEL:
            span_file_name = "/test_span_max_model.pth" if type_ == "test" else "/dev_span_max_model.pth"
            span_best_trees_parsed = "/test_span_trees.pkl" if type_ == "test" else "/dev_span_trees.pkl"
            ns_file_name = "/test_nucl_max_model.pth" if type_ == "test" else "/dev_nucl_max_model.pth"
            ns_best_trees_parsed = "/test_ns_trees.pkl" if type_ == "test" else "/dev_ns_trees.pkl"
            rel_file_name = "/test_rel_max_model.pth" if type_ == "test" else "/dev_rel_max_model.pth"
            rel_best_trees_parsed = "/test_rel_trees.pkl" if type_ == "test" else "/dev_rel_trees.pkl"
            f_file_name = "/test_f_max_model.pth" if type_ == "test" else "/dev_f_max_model.pth"
            f_best_trees_parsed = "/test_f_trees.pkl" if type_ == "test" else "/dev_f_trees.pkl"
            if span_pre[1] > span_max[1]:
                self.save_model(file_name=span_file_name, model=model)
                self.save_trees(file_name=span_best_trees_parsed, trees=predtrees)
            if nucl_pre[1] > nucl_max[1]:
                self.save_model(file_name=ns_file_name, model=model)
                self.save_trees(file_name=ns_best_trees_parsed, trees=predtrees)
            if rel_pre[1] > rel_max[1]:
                self.save_model(file_name=rel_file_name, model=model)
                self.save_trees(file_name=rel_best_trees_parsed, trees=predtrees)
            if f_pre[1] > f_max[1]:
                self.save_model(file_name=f_file_name, model=model)
                self.save_trees(file_name=f_best_trees_parsed, trees=predtrees)

    @staticmethod
    def save_model(file_name, model):
        # 存储
        dir2save = MODELS2SAVE + "/v" + str(VERSION) + "_set" + str(SET_of_version)
        safe_mkdir(MODELS2SAVE)
        safe_mkdir(dir2save)
        save_path = dir2save + file_name
        torch.save(model, save_path)

    @staticmethod
    def save_trees(file_name, trees):
        # 存储
        dir2save = MODELS2SAVE + "/v" + str(VERSION) + "_set" + str(SET_of_version)
        safe_mkdir(MODELS2SAVE)
        safe_mkdir(dir2save)
        save_path = dir2save + file_name
        save_data(trees, save_path)

    def output_report(self, report_info=None):
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
        """ (mi, inner_mi, ma, inner_ma)
        """
        report_info = []
        if CROSS_VAL:
            report_info.append("==================== DEV =====================")
            report_info.append("(Micro) S: " + str(self.dev_span_max[0]) + " -- N: " +
                               str(self.dev_nucl_max[0]) + " -- R:" + str(self.dev_rel_max[0])
                               + " -- F:" + str(self.dev_f_max[0]))
            report_info.append("(Macro) S: " + str(self.dev_span_max[2]) + " -- N: " +
                               str(self.dev_nucl_max[2]) + " -- R:" + str(self.dev_rel_max[2])
                               + " -- F:" + str(self.dev_f_max[2]))
            report_info.append("(Micro_Inner) S: " + str(self.dev_span_max[1]) + " -- N: " + str(self.dev_nucl_max[1])
                               + " -- R:" + str(self.dev_rel_max[1]) + " -- F:" + str(self.dev_f_max[1]))
            report_info.append("(Macro_Inner) S: " + str(self.dev_span_max[3]) + " -- N: " + str(self.dev_nucl_max[3])
                               + " -- R:" + str(self.dev_rel_max[3]) + " -- F:" + str(self.dev_f_max[3]))
        report_info.append("==================== TEST =====================")
        report_info.append("(Micro) S: " + str(self.test_span_max[0]) + " -- N: " +
                           str(self.test_nucl_max[0]) + " -- R:" + str(self.test_rel_max[0]) +
                           " -- F:" + str(self.test_f_max[0]))
        report_info.append("(Macro) S: " + str(self.test_span_max[2]) + " -- N: " +
                           str(self.test_nucl_max[2]) + " -- R:" + str(self.test_rel_max[2]) +
                           " -- F:" + str(self.test_f_max[2]))
        report_info.append("(Micro_Inner) S: " + str(self.test_span_max[1]) + " -- N: " + str(self.test_nucl_max[1])
                           + " -- R:" + str(self.test_rel_max[1]) + " -- F:" + str(self.test_f_max[1]))
        report_info.append("(Macro_Inner) S: " + str(self.test_span_max[3]) + " -- N: " + str(self.test_nucl_max[3])
                           + " -- R:" + str(self.test_rel_max[3]) + " -- F:" + str(self.test_f_max[3]))
        return report_info
