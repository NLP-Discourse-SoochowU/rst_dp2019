# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date:
@Description: 对EDU编码器的整理
              attn_buffer在对EDU编码时用Bi-LSTM计算得到的隐层，存储。
              attn_cache在对spinn模型进行forward时一步一步对area区域的attn结果的attention计算。
"""
import torch
import numpy as np
from config import *
import torch.nn as nn
from collections import deque
from sklearn.decomposition import PCA
from torch.nn import functional as nnfunc
from torch.autograd import Variable as Var
torch.manual_seed(SEED)


class edu_encoder:
    def __init__(self, word2ids, pos2ids, wordemb):
        self.word2ids = word2ids
        self.wordemb = wordemb
        self.pos2ids = pos2ids
        self.posemb = nn.Embedding(len(pos2ids.keys()), POS_EMBED_SIZE)  # 随机初始化pos_embed
        # projections
        self.edu_proj = nn.Linear(EMBED_SIZE * 2 + POS_EMBED_SIZE + 2 * SPINN_HIDDEN, 2 * SPINN_HIDDEN)

        # gated_model
        self.attn_buffer = deque()  # 存储edus_attention的初始情况
        self.attn_cache = deque()  # 存储自底向上的过程

        # self attention
        edu_rnn_encoder_size = 2 * SPINN_HIDDEN
        self.edu_rnn_encoder = nn.LSTM(EMBED_SIZE + POS_EMBED_SIZE, edu_rnn_encoder_size // 2, bidirectional=True)
        self.edu_gru_encoder = nn.GRU(EMBED_SIZE + POS_EMBED_SIZE, edu_rnn_encoder_size // 2, bidirectional=True)
        self.span_rnn_encoder = nn.LSTM(edu_rnn_encoder_size, edu_rnn_encoder_size // 2, bidirectional=True)
        self.edu_attn_query = nn.Parameter(torch.randn(edu_rnn_encoder_size))
        self.edu_attn = nn.Sequential(
            nn.Linear(edu_rnn_encoder_size, edu_rnn_encoder_size, bias=False),
            nn.Tanh()
        )

        # attention for area attention, 注意这里的输入即为各个edu的编码隐藏层即为 2*hidden_size或者无监督的EMBED_SIZE
        area_rnn_encoder_size = EMBED_SIZE
        self.area_rnn_encoder = nn.LSTM(area_rnn_encoder_size, area_rnn_encoder_size)
        self.area_attn_query = nn.Parameter(torch.randn(area_rnn_encoder_size))
        self.area_attn = nn.Sequential(
            nn.Linear(area_rnn_encoder_size, area_rnn_encoder_size),
            nn.Tanh()
        )
        self.area_tmp_attn_vec = None  # 当前area的attn结果向量
        self.proj_dropout = nn.Dropout(proj_dropout)

    def edu_bilstm_encode(self, word_emb, tags_emb):
        """ 对 edu 双向 lstm 进行编码
        """
        inputs = torch.cat([word_emb, tags_emb], 1).unsqueeze(1)  # (seq_len, batch, input_size)
        hs, _ = self.edu_rnn_encoder(inputs)  # hs.size()  (seq_len, batch, hidden_size)
        hs = hs.squeeze()  # size: (seq_len, hidden_size)
        keys = self.edu_attn(hs)  # size: (seq_len, hidden_size)
        attn = nnfunc.softmax(keys.matmul(self.edu_attn_query), 0)
        output = (hs * attn.view(-1, 1)).sum(0)
        return output, attn

    def get_words_tag_ids(self, edu_ids, pos_ids):
        """ 获取词等标签的 ids
        """
        if len(edu_ids) == 1:
            w1 = edu_ids[0]
            w2 = self.word2ids[PAD]
            w3 = self.word2ids[PAD]
            w_1 = self.word2ids[PAD]
            p1 = pos_ids[0]
            p2 = self.pos2ids[PAD]
            p3 = self.pos2ids[PAD]
            p_1 = self.pos2ids[PAD]
        elif len(edu_ids) == 2:
            w1 = edu_ids[0]
            w2 = edu_ids[1]
            w3 = self.word2ids[PAD]
            w_1 = self.word2ids[PAD]
            p1 = pos_ids[0]
            p2 = pos_ids[1]
            p3 = self.pos2ids[PAD]
            p_1 = self.pos2ids[PAD]
        else:
            w1 = edu_ids[0]
            w2 = edu_ids[1]
            w3 = edu_ids[2]
            w_1 = edu_ids[-1]
            p1 = pos_ids[0]
            p2 = pos_ids[1]
            p3 = pos_ids[2]
            p_1 = pos_ids[-1]
        return [w1, w2, w3, w_1], [p1, p2, p3, p_1]

    @staticmethod
    def pad_edu(edu_ids=None, pos_ids=None):
        """ 对 edu 进行 pad
        """
        edu_ids_list = edu_ids[:]
        pos_ids_list = pos_ids[:] if pos_ids is not None else []
        while len(edu_ids_list) < PAD_SIZE:
            edu_ids_list = np.append(edu_ids_list, PAD_ids)
            if pos_ids is not None:
                pos_ids_list = np.append(pos_ids_list, PAD_ids)
        padded_val = Var(torch.LongTensor(edu_ids_list)), Var(torch.LongTensor(pos_ids_list)) if \
            pos_ids is not None else Var(torch.LongTensor(edu_ids_list))
        return padded_val

    def edu_encode(self, edu):
        """ Use the 0 1 -1 word vector of a sentence to encode an EDU
            Input: An object of rst_tree, leaf node
            Output: An output of code with lower dimension.
        """
        edu_ids = edu.temp_edu_ids[:]
        pos_ids = edu.temp_pos_ids[:]
        edu_elmo = edu.temp_edu_emlo_emb[:] if USE_ELMo else None
        if len(edu_ids) == 0:
            return torch.zeros(SPINN_HIDDEN * 2)
        # encode EDUs with Bi-LSTM, average them
        word_emb = edu_elmo if USE_ELMo else self.wordemb(torch.LongTensor(edu_ids))  # (50, 100)
        pos_embed = self.posemb(torch.LongTensor(pos_ids))
        hs = self.edu_rnn_encoder(torch.cat([word_emb, pos_embed], 1).unsqueeze(1))[0].squeeze(1)
        rnn_emb = torch.mean(hs, 0)
        # rnn_emb, attn = self.edu_bilstm_encode(word_emb, pos_embed)
        edu_embed = torch.cat([word_emb[0], word_emb[-1], pos_embed[0], rnn_emb]) if USE_W_P_IN_BI_LSTM else rnn_emb
        proj_out = self.edu_proj(edu_embed) if USE_W_P_IN_BI_LSTM else edu_embed
        return self.proj_dropout(proj_out)

    def area_attn_encode(self):
        """ 对当前区域的所有 edu 的 attentions 进行 attn 获取并计算当前区域的注意力的注意力
            area_edus_att: shape = (num of edu, 2 * SPINN_HIDDEN)
        """
        tmp_area_info = self.attn_cache[-1]
        # 是否数值累加反而能很好地表征群体本身
        self.area_tmp_attn_vec = torch.mean(tmp_area_info, 0)

    def edu2vec_unsupervised(self, edu_list, a: float = 1e-3):
        """ A SIMPLE BUT TOUGH TO BEAT BASELINE FOR SENTENCE EMBEDDINGS, Princeton University
            convert a list of sentence with word2vec items into a set of sentence vectors
            :param edu_list: EDU列表
            :param a: 超参
        """
        sentences = None
        for edu in edu_list:
            # add all word2vec values into one vector for the sentence
            vs = Var(torch.zeros(EMBED_SIZE))
            for word, freq in zip(edu.temp_edu_ids, edu.temp_edu_freq):
                a_value = torch.tensor(a / (a + freq))  # smooth inverse frequency, SIF
                vs = vs + a_value * self.wordemb(torch.tensor(word))
                # vs += sif * word_vector
            vs = torch.div(vs, torch.tensor(float(len(edu.temp_edu_ids))))
            vs = torch.unsqueeze(vs, 0)
            sentences = vs if sentences is None else torch.cat((sentences, vs), 0)

        # 对sentence set进行主成分分析（降维）
        pca = PCA(n_components=EMBED_SIZE)
        pca.fit(sentences.data.numpy())  # np.array(sentences)
        # the PCA vector，不需要这里更新，这是无监督计算方式
        u = pca.components_[0]
        u = np.multiply(u, np.transpose(u))  # uuT
        # 当u小于embedding_size需要pad
        if len(u) < EMBED_SIZE:
            for i in range(EMBED_SIZE - len(u)):
                u = np.append(u, 0)  # add needed extension for multiplication below
        # resulting sentence vectors: vs = vs -u x uT x vs 与论文中一致
        for vs in sentences:
            sub = torch.mul(torch.FloatTensor(u), vs)
            self.attn_buffer.append(torch.sub(vs, sub))

    def edu2vec_unsupervised_origin(self, edu_list, a: float = 1e-3):
        """ A SIMPLE BUT TOUGH TO BEAT BASELINE FOR SENTENCE EMBEDDINGS, Princeton University
            convert a list of sentence with word2vec items into a set of sentence vectors
            :param edu_list: EDU列表
            :param a: 超参
        """
        sentence_list = []
        for edu in edu_list:
            # add all word2vec values into one vector for the sentence
            vs = np.zeros(EMBED_SIZE)
            if USE_ELMo:
                for word_emb, freq in zip(edu.temp_edu_emlo_emb, edu.temp_edu_freq):
                    a_value = a / (a + freq)  # smooth inverse frequency, SIF
                    vs = np.add(vs, np.multiply(a_value, word_emb.data.numpy()))
            else:
                for word, freq in zip(edu.temp_edu_ids, edu.temp_edu_freq):
                    a_value = a / (a + freq)  # smooth inverse frequency, SIF
                    vs = np.add(vs, np.multiply(a_value, self.wordemb(torch.tensor(word)).data.numpy()))
                    # vs += sif * word_vector
            vs = np.divide(vs, len(edu.temp_edu_ids))
            sentence_list.append(vs)

        # 对sentence set进行主成分分析（降维）
        pca = PCA(n_components=EMBED_SIZE)
        pca.fit(np.array(sentence_list))
        # the PCA vector
        u = pca.components_[0]
        u = np.multiply(u, np.transpose(u))  # uuT

        # 当u小于embedding_size需要pad
        if len(u) < EMBED_SIZE:
            for i in range(EMBED_SIZE - len(u)):
                u = np.append(u, 0)  # add needed extension for multiplication below
        # resulting sentence vectors: vs = vs -u x uT x vs 与论文中一致
        for vs in sentence_list:
            sub = np.multiply(u, vs)  # u = u u^T, vs len(word_vec) 300, 300, 300
            self.attn_buffer.append(np.subtract(vs, sub))
