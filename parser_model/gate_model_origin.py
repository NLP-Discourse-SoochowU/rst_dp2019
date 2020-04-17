# utf-8

"""
    Author: Lyzhang
    Date: 2018.8.15
    Description:
"""
import torch
import torch.nn as nn
from torch.nn import functional as nnfunc
from config import *
torch.manual_seed(SEED)


class tri_gate_model(nn.Module):
    def __init__(self):
        super(tri_gate_model, self).__init__()
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
        angle_out_all = None
        angle_prop_all = None
        if GATE_ANGLE_NUM == 1:
            hidden_states = hidden_states.view(-1, SPINN_HIDDEN)
            str_info = self.left_linear(hidden_states[0]) + self.right_linear(hidden_states[1]).unsqueeze(0)
            track_info = self.tracking_linear(tracking_vec).unsqueeze(0)
            area_info = self.area_linear(area_att_vec).unsqueeze(0)
            info_flows = torch.cat((track_info, area_info, str_info), 0)
            scores = self.looker(self.gate_dropout(info_flows)).view(-1)
            angle_prop_all = nnfunc.softmax(scores, 0).view(-1, 1)
            angle_out_all = (info_flows * angle_prop_all).view(-1)
        else:
            for angle_idx in range(GATE_ANGLE_NUM):
                scores = torch.cat((self.track2single[angle_idx](tracking_vec),
                                    self.area2single[angle_idx](area_att_vec),
                                    self.hidden2single[angle_idx](hidden_states)))
                prop = nnfunc.softmax(scores, 0).view(-1, 1)
                # (3, 2 * HIDDEN_SIZE)
                output = torch.cat((torch.mul(tracking_vec, prop[0]),
                                    torch.mul(area_att_vec, prop[1]),
                                    torch.mul(hidden_states, prop[2])))
                angle_prop_all = prop if angle_out_all is None else torch.cat((angle_prop_all, prop), 1)
                angle_out_all = output if angle_out_all is None else torch.cat((angle_out_all, output))
        return angle_out_all, angle_prop_all
