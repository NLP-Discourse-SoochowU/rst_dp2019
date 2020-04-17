# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date:
@Description:
"""
import torch
import torch.nn as nn
from config import SEED, Tracking_With_GRU
from utils.file_util import load_data
from config import ALL_LABELS_NUM, Action2ids_path, LABEL_EMBED_SIZE


class Tracker(nn.Module):
    """ Desc: tracker for tree lstm
    """
    def __init__(self, hidden_size):
        nn.Module.__init__(self)
        torch.manual_seed(SEED)
        self.hidden_size = hidden_size
        input_size = 3 * self.hidden_size
        self.rnn = nn.LSTMCell(input_size, hidden_size)
        self.gru = nn.GRUCell(input_size, hidden_size)
        self.label_emb = nn.Embedding(ALL_LABELS_NUM, LABEL_EMBED_SIZE)
        self.label_emb.requires_grad = True
        self.init_label = nn.Parameter(torch.randn(LABEL_EMBED_SIZE))  # 初始状态的上一个操作未知
        self.unk_label = nn.Parameter(torch.randn(LABEL_EMBED_SIZE))  # 初始状态的上一个操作未知
        self.label2ids = load_data(Action2ids_path)

    def forward(self, stack, buffer_, state, label=None):
        s1, s2 = stack[-1], stack[-2]
        b1 = buffer_[0]
        s2h, s2c = s2.chunk(2)
        s1h, s1c = s1.chunk(2)
        b1h, b1c = b1.chunk(2)
        cell_input = (torch.cat([s2h, s1h, b1h])).view(1, -1)
        # state1, state2 = state
        if Tracking_With_GRU:
            tracking_h = self.gru(cell_input, state)
            tracking_out = tracking_h.view(1, -1)
        else:
            tracking_h, tracking_c = self.rnn(cell_input, state)
            tracking_out = tracking_h.view(1, -1), tracking_c.view(1, -1)
        return tracking_out
