# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date:
@Description:
"""
import torch
from config import *
import torch.nn as nn
from parser_model.gate_model import gate_model, bi_gate_model, tri_gate_model
torch.manual_seed(SEED)


class Reducer(nn.Module):
    """ Desc： The composition function for reduce option.
    """
    def __init__(self, hidden_size):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        # 为特殊的gated_model创建线性变换
        # self.proj_SENT2VEC = nn.Linear(self.hidden_size * 2 + EMBED_SIZE, self.hidden_size * 5)
        # self.proj_2H = nn.Sequential(
        #     nn.Linear(self.hidden_size * 2 * GATE_ANGLE_NUM, self.hidden_size * 5),
        #     nn.Dropout(p=mlp_dropout)  # 当前概率为0
        #     )
        # self.proj_3H = nn.Sequential(
        #     nn.Linear(self.hidden_size * 3 * GATE_ANGLE_NUM, self.hidden_size * 5),
        #     nn.Dropout(p=mlp_dropout)
        #     )
        # self.proj_4H = nn.Sequential(
        #     nn.Linear(self.hidden_size * 4 * GATE_ANGLE_NUM, self.hidden_size * 5),
        #     nn.Dropout(p=mlp_dropout)
        #     )
        # self.proj_5H = nn.Sequential(
        #     nn.Linear(self.hidden_size * 5 * GATE_ANGLE_NUM, self.hidden_size * 5),
        #     nn.Dropout(p=mlp_dropout)
        # )
        # self.proj_6H = nn.Sequential(
        #     nn.Linear(self.hidden_size * 6 * GATE_ANGLE_NUM, self.hidden_size * 5),
        #     nn.Dropout(p=mlp_dropout)
        # )
        # self.proj_2H_UNS = nn.Sequential(
        #     nn.Linear((self.hidden_size * 2 + EMBED_SIZE) * GATE_ANGLE_NUM, self.hidden_size * 5),
        #     nn.Dropout(p=mlp_dropout)
        # )
        # self.proj_3H_UNS = nn.Sequential(
        #     nn.Linear((self.hidden_size * 3 + EMBED_SIZE) * GATE_ANGLE_NUM, self.hidden_size * 5),
        #     nn.Dropout(p=mlp_dropout)
        # )
        self.projection = nn.Sequential(
            nn.Linear((self.hidden_size * 3 + EMBED_SIZE) * GATE_ANGLE_NUM, self.hidden_size * 5),
            nn.Dropout(p=mlp_dropout)
        )
        self.gate = gate_model()
        self.super_bi_gate = bi_gate_model()
        self.tri_gate = tri_gate_model()
        self.drop = nn.Dropout(p=mlp_dropout)

    # def get_out_(self, gated_output):
    #     output_size = gated_output.size()[0]
    #     input(output_size)
    #     if output_size == 2 * SPINN_HIDDEN * GATE_ANGLE_NUM:
    #         g, i, f1, f2, o = self.proj_2H(gated_output).chunk(5)
    #     elif output_size == 3 * SPINN_HIDDEN * GATE_ANGLE_NUM:
    #         g, i, f1, f2, o = self.proj_3H(gated_output).chunk(5)
    #     elif output_size == 4 * SPINN_HIDDEN * GATE_ANGLE_NUM:
    #         g, i, f1, f2, o = self.proj_4H(gated_output).chunk(5)
    #     elif output_size == (2 * SPINN_HIDDEN + EMBED_SIZE) * GATE_ANGLE_NUM:
    #         g, i, f1, f2, o = self.proj_2H_UNS(gated_output).chunk(5)
    #     elif output_size == (3 * SPINN_HIDDEN + EMBED_SIZE) * GATE_ANGLE_NUM:
    #         g, i, f1, f2, o = self.proj_3H_UNS(gated_output).chunk(5)
    #     elif output_size == (4 * SPINN_HIDDEN + EMBED_SIZE) * GATE_ANGLE_NUM:
    #         g, i, f1, f2, o = self.proj_4H_UNS(gated_output).chunk(5)
    #     elif output_size == 5 * SPINN_HIDDEN * GATE_ANGLE_NUM:
    #         g, i, f1, f2, o = self.proj_5H(gated_output).chunk(5)
    #     elif output_size == 6 * SPINN_HIDDEN * GATE_ANGLE_NUM:
    #         g, i, f1, f2, o = self.proj_6H(gated_output).chunk(5)
    #     else:
    #         print("文件reducer.py")
    #         input(output_size)
    #         g, i, f1, f2, o = 0, 0, 0, 0, 0
    #     return g, i, f1, f2, o

    def forward(self, left, right, tracking, area_attn_evc):
        """ Desc:   The forward of Reducer
            input:  The rep of left node and right node, e is the tree lstm's output, it has a different D.
            output: The rep of temp node
            :param area_attn_evc: shape = (num of edu, 2 * SPINN_HIDDEN) or (num of edu, embed_size)
        """
        angle_prop_all = None
        h1, c1 = left.chunk(2)
        h2, c2 = right.chunk(2)
        e_h = tracking if Tracking_With_GRU else tracking[0]
        e_h = e_h.squeeze()
        tracking_vector = e_h
        hidden_states = torch.cat((h1, h2))
        gated_output, angle_prop_all = self.tri_gate(tracking_vector, area_attn_evc, hidden_states)
        angle_prop_all = angle_prop_all.permute(1, 0) if GATE_ANGLE_NUM > 1 and angle_prop_all is not None \
            else angle_prop_all
        g, i, f1, f2, o = self.projection(gated_output).chunk(5)
        c = g.tanh() * i.sigmoid() + f1.sigmoid() * c1 + f2.sigmoid() * c2
        h = o.sigmoid() * c.tanh()
        return torch.cat([h, c]), angle_prop_all
