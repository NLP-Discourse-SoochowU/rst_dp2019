# utf-8

"""
    Author: Lyzhang
    Date: 2018.8.15
    Description: 对输入的 Tracking vector 和 Area attention vector 作为互斥门控的输入信息，用 α 和 1-α 来控制两个向量的输入。
"""
import torch
import torch.nn as nn
from torch.nn import functional as nnfunc
from config import *
torch.manual_seed(SEED)


class tri_gate_model(nn.Module):
    """
    对三个输入源进行联合控制β, γ and (1-β-γ) 我只根据降维的结果计算注意力信息，但是还是保留原来信息进行计算
    结合多角度+互斥门控机制
        # self.attn_query = nn.Parameter(torch.randn(2*SPINN_HIDDEN))
        # self.uns_edu_trans = nn.Linear(EMBED_SIZE, 2*SPINN_HIDDEN)
        # self.tracking_2_Bi_LSTM_SIZE = nn.Linear(SPINN_HIDDEN, 2*SPINN_HIDDEN)
        # self.info2single = nn.Linear(2*SPINN_HIDDEN, SPINN_HIDDEN)
    """
    def __init__(self):
        super(tri_gate_model, self).__init__()
        # 概率计算参数
        area_in_size = EMBED_SIZE
        track_in_size = SPINN_HIDDEN
        self.area2single = nn.Linear(area_in_size, 1) if GATE_ANGLE_NUM == 1 else []
        self.track2single = nn.Linear(track_in_size, 1) if GATE_ANGLE_NUM == 1 else []
        self.hidden2single = nn.Linear(2 * SPINN_HIDDEN, 1) if GATE_ANGLE_NUM == 1 else []
        # Linear is necessary
        self.area_linear = nn.Linear(area_in_size, SPINN_HIDDEN)
        self.tracking_linear = nn.Linear(track_in_size, SPINN_HIDDEN)
        self.left_linear = nn.Linear(SPINN_HIDDEN, SPINN_HIDDEN)
        self.right_linear = nn.Linear(SPINN_HIDDEN, SPINN_HIDDEN)
        self.looker = nn.Linear(SPINN_HIDDEN, 1)
        # drop out
        self.gate_dropout = nn.Dropout(gate_drop_out_rate)
        self.a_norm = nn.LayerNorm(track_in_size)
        self.hidden_norm = nn.LayerNorm(2 * SPINN_HIDDEN)

    def forward(self, tracking_vec, area_att_vec, hidden_states):
        """
        这里讲对三个关键信息进行门控，要求控制关闭一些门的操作在这里直接根据属性 Gate_Close_V 控制
        注意，这里需要对三种信息进行变换: 当area大小为embed_size的时候，当tracking_vec不包含conn的时候，如何求概率
        :param tracking_vec:
        :param area_att_vec:
        :param hidden_states:
        :return:
        """
        angle_out_all = None
        angle_prop_all = None
        if GATE_ANGLE_NUM == 1:
            hidden_states = hidden_states.view(-1, SPINN_HIDDEN)
            str_info = self.left_linear(hidden_states[0]) + self.right_linear(hidden_states[1]).unsqueeze(0)
            track_info = self.tracking_linear(tracking_vec).unsqueeze(0)
            area_info = self.area_linear(area_att_vec).unsqueeze(0)
            info_flows = torch.cat((track_info, area_info, str_info), 0)
            scores = self.looker(self.gate_dropout(info_flows)).view(-1)
            # tracking信息 + Area主成分信息 + str结构信息
            angle_prop_all = nnfunc.softmax(scores, 0).view(-1, 1)
            angle_out_all = (info_flows * angle_prop_all).view(-1)
        else:
            # 多角度分配过程
            for angle_idx in range(GATE_ANGLE_NUM):
                scores = torch.cat((self.track2single[angle_idx](tracking_vec),
                                    self.area2single[angle_idx](area_att_vec),
                                    self.hidden2single[angle_idx](hidden_states)))
                prop = nnfunc.softmax(scores, 0).view(-1, 1)
                # (3, 2*HIDDEN_SIZE)
                output = torch.cat((torch.mul(tracking_vec, prop[0]),
                                    torch.mul(area_att_vec, prop[1]),
                                    torch.mul(hidden_states, prop[2])))
                angle_prop_all = prop if angle_out_all is None else torch.cat((angle_prop_all, prop), 1)
                angle_out_all = output if angle_out_all is None else torch.cat((angle_out_all, output))
        return angle_out_all, angle_prop_all
