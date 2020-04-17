# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date: 2018/5/4
@Description:
"""
import torch
from config import *
import torch.nn as nn
from collections import deque
from parser_model.mlp import MLP
from parser_model.reducer import Reducer
from parser_model.trackers import Tracker
from torch.autograd import Variable as Var
from parser_model.edu_encoder import edu_encoder
torch.manual_seed(SEED)


class SPINN(nn.Module):
    def __init__(self, word2ids, pos2ids, wordemb_weights):
        nn.Module.__init__(self)
        # word embedding
        self.wordemb = nn.Embedding(len(word2ids), 100)  # GloVe
        self.wordemb.weight.data.copy_(torch.from_numpy(wordemb_weights))
        self.wordemb.weight.requires_grad = False
        
        self.edu_encoder = edu_encoder(word2ids, pos2ids, self.wordemb)
        # two trackers
        self.tracker = Tracker(SPINN_HIDDEN)
        self.reducer = Reducer(SPINN_HIDDEN)
        # mlp
        tran_in_size = mlp_input_size
        nucl_in_size = area_input_size
        rel_in_size = area_input_size
        self.mlp_tran = MLP(input_size=tran_in_size, output_size=Transition_num, num_layers=MLP_LAYERS)
        self.mlp_nucl = MLP(input_size=nucl_in_size, output_size=NUCL_NUM, num_layers=MLP_LAYERS)
        self.mlp_rel = MLP(input_size=rel_in_size, output_size=COARSE_REL_NUM, num_layers=MLP_LAYERS)
        self.mlp_nr = MLP(input_size=area_input_size, output_size=NR_NUM, num_layers=MLP_LAYERS)
        # hidden_tran
        self.hidden_tran = nn.Linear(EMBED_SIZE, 2 * SPINN_HIDDEN)

    @staticmethod
    def copy_session(session):
        """ Desc: return a copy of a session.
        """
        stack_, buffer_, tracking = session
        stack_clone = [s.clone() for s in stack_]
        buffer_clone = deque([b.clone() for b in buffer_])
        h, c = tracking
        tracking_clone = h.clone(), c.clone()
        return stack_clone, buffer_clone, tracking_clone

    def new_session(self, tree):
        """ Desc: Create a new session  内存占用问题待解决
            Input: the root of a new tree
            Output: stack_, buffer, tracking
        """
        # 初始状态空栈中存在两个空数据
        stack_ = [Var(torch.zeros(SPINN_HIDDEN * 2)) for _ in range(2)]  # [dumb, dumb]
        # 初始化队列
        buffer_ = deque()
        self.edu_encoder.attn_cache = deque()  # 在下面对所有edus迭代编码的过程中对当前篇章的各个edu的attention获取并存储到缓冲区中
        # 计算无监督方式得到的各个EDU的表示
        self.edu_encoder.edu2vec_unsupervised_origin(tree.edus)

        for edu_ in tree.edus:
            buffer_.append(self.edu_encoder.edu_encode(edu_))  # 对edu进行编码
        buffer_.append(Var(torch.zeros(SPINN_HIDDEN * 2)))  # [edu, edu, ..., dumb]
        tracker_init_state = Var(torch.zeros(1, SPINN_HIDDEN)) if Tracking_With_GRU else \
            (Var(torch.zeros(1, SPINN_HIDDEN)), Var(torch.zeros(1, SPINN_HIDDEN)))
        tracking = self.tracker(stack_, buffer_, tracker_init_state)  # forward of Tracker
        return stack_, buffer_, tracking

    def score_tran(self, session):
        """ Desc: sigmoid(fullc(h->1))
            使用BCE loss的时候返回一个概率，用sigmoid
            使用Cross entropy loss的时候返回一组概率值，个数和标签数一致
        """
        stack_, buffer_, tracking = session
        h = tracking if Tracking_With_GRU else tracking[0]
        score_h = h
        score_output = self.mlp_tran(score_h)  # (1, 584)
        return score_output

    def score_nucl(self, session):
        """ Desc: sigmoid(fullc(h->1))
            使用BCE loss的时候返回一个概率，用sigmoid
            使用Cross entropy loss的时候返回一组概率值，个数和标签数一致
        """
        stack_, buffer_, tracking = session
        h = tracking if Tracking_With_GRU else tracking[0]
        # 将左右孩子的信息加入到转移预测中
        right_child = stack_[-1].unsqueeze(0)
        left_child = stack_[-2].unsqueeze(0)
        score_h = torch.cat((h, right_child, left_child), 1)
        score_output = self.mlp_nucl(score_h)  # (1, 584)
        return score_output

    def score_rel(self, session):
        """ Desc: sigmoid(fullc(h->1))
            使用BCE loss的时候返回一个概率，用sigmoid
            使用Cross entropy loss的时候返回一组概率值，个数和标签数一致
            注意：只有当执行reduce的时候才会执行关系标签的预测，那么这里可以仅使用左右孩子的表征进行关系预测，至于特征再说
        """
        stack_, buffer_, tracking = session
        h = tracking if Tracking_With_GRU else tracking[0]
        # 考虑对于关系预测更多是关于area之间的关系，可能和转移序列之间的关系并不是太大，所以考虑改变
        right_child = stack_[-1].unsqueeze(0)
        left_child = stack_[-2].unsqueeze(0)
        score_h = torch.cat((h, right_child, left_child), 1)
        score_output = self.mlp_rel(score_h)
        return score_output

    def score_nr(self, session):
        stack_, buffer_, tracking = session
        h = tracking if Tracking_With_GRU else tracking[0]
        right_child = stack_[-1].unsqueeze(0)
        left_child = stack_[-2].unsqueeze(0)
        score_h = torch.cat((h, right_child, left_child), 1)
        score_output = self.mlp_nr(score_h)
        return score_output

    def forward(self, session, transition):
        """ Desc: The forward of SPINN
            Input: session and (shift or reduce)
            output: newest stack and buffer, lstm output
        """
        angle_prop_all = None
        stack_, buffer_, tracking = session
        if transition == SHIFT:
            stack_.append(buffer_.popleft())
            tmp_dt = torch.FloatTensor(self.edu_encoder.attn_buffer.popleft())
            self.edu_encoder.attn_cache.append(tmp_dt.unsqueeze(0))
        else:
            transition = transition[0] + "-" + transition[1]
            s1 = stack_.pop()
            s2 = stack_.pop()
            # area_attn get
            area_attn_r = self.edu_encoder.attn_cache.pop()
            area_attn_l = self.edu_encoder.attn_cache.pop()
            tmp_area_attn = torch.cat((area_attn_l, area_attn_r), 0)
            self.edu_encoder.attn_cache.append(tmp_area_attn)
            self.edu_encoder.area_attn_encode()  # 计算当前区域的attn结果
            compose, angle_prop_all = self.reducer(s2, s1, tracking, self.edu_encoder.area_tmp_attn_vec)
            # the forward of Reducer
            stack_.append(compose)
        # 最新状态转移
        tracking = self.tracker(stack_, buffer_, tracking, label=transition)  # The forward of the Tracker
        session_ = stack_, buffer_, tracking
        return session_, angle_prop_all
