# -*- coding: utf-8 -*-

"""
@Author: lyzhang
@Date:
@Description: count the frequency of words in the corpus WSJ, all articles
"""
import os
from utils.file_util import save_data, load_data
from utils.text_process_util import get_sent_words_syns
from path_config import WORD_FREQUENCY, VOC_WORD2IDS_PATH, WSJ_PATH


def freq_count():
    word_c = 0
    word_freq_dict = dict()
    for folder_name in os.listdir(WSJ_PATH):
        folder_path = os.path.join(WSJ_PATH, folder_name)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            print(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0 or line.strip() == ".START":
                        continue
                    print(file_path)
                    print(line)
                    words = get_sent_words_syns(line)
                    word_c += len(words)
                    for word in words:
                        word_freq_dict[word] = word_freq_dict[word] + 1 if word in word_freq_dict.keys() else 1
                    print("over")
    return word_freq_dict, word_c


def freq2rate(word_freq_dict, word_all, word2ids):
    word_all = float(word_all)
    ids2rate = dict()
    for key, val in zip(word_freq_dict.keys(), word_freq_dict.values()):
        if key in word2ids.keys():
            ids2rate[word2ids[key]] = val / word_all
    # id 检测
    unk_counter = 0
    for ids in word2ids.values():
        if ids not in ids2rate.keys():
            ids2rate[ids] = 1 / word_all
            unk_counter += 1
    print("UNK_COUNT: ", unk_counter)
    save_data(ids2rate, WORD_FREQUENCY)


def print_freq_and_word(word2f, word_count):
    result = sorted(word2f.items(), key=lambda item: item[1], reverse=True)
    print("count all ", word_count, ".")
    for item in result:
        input(item)


if __name__ == "__main__":
    word2freq, word_counter = freq_count()
    word_to_ids = load_data(VOC_WORD2IDS_PATH)
    freq2rate(word2freq, word_counter, word_to_ids)
    print_freq_and_word(word2freq, word_counter)
