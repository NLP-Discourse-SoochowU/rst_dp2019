# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date:
@Description:
"""
import torch
from config import *
import torch.nn as nn
torch.manual_seed(SEED)


class MLP(nn.Module):
    def __init__(self, input_size=None, output_size=None, hidden_size=None, num_layers=None, use_sig=True):
        nn.Module.__init__(self)
        input_size = input_size if input_size is not None else mlp_input_size
        hidden_size = hidden_size if hidden_size is not None else int(input_size / 2)
        num_layers = num_layers if num_layers is not None else MLP_LAYERS
        output_size = output_size if output_size is not None else Transition_num
        self.use_sig = use_sig

        # part one
        # linear probability
        self.linear_logits = nn.Linear(input_size, output_size)

        # part two
        # input to first hidden
        self.input_linear = nn.Linear(input_size, hidden_size)
        self.input_dropout = nn.Dropout(p=mlp_dropout)
        self.input_activation = nn.ReLU()

        # multi hidden layers
        self.linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_layers - 2)])
        self.dropouts = nn.ModuleList([nn.Dropout(p=mlp_dropout) for _ in range(num_layers - 2)])
        self.activations = nn.ModuleList([nn.ReLU() for _ in range(num_layers - 2)])

        # probabilities
        self.logits = nn.Linear(hidden_size, output_size)

    def forward(self, input_values):
        if MLP_LAYERS == 1:
            output = self.linear_logits(input_values)
        else:
            hidden = self.input_linear(input_values)
            hidden = self.input_dropout(hidden)
            hidden = self.input_activation(hidden)
            # hidden layers
            for linear, dropout, activation in zip(self.linears, self.dropouts, self.activations):
                hidden = linear(hidden)
                hidden = dropout(hidden)
                hidden = activation(hidden)
            # sigmoid output layer
            output = self.logits(hidden)
        sig_output = output if not self.use_sig else torch.tanh(output)
        return sig_output
