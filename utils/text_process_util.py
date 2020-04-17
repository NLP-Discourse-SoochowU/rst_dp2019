"""
    作者：张龙印
    日期：2018.3.12
    文本处理
    文本处理
"""
import re
# 定义英文符号
s_list = [',', '.', ':', ';', '!', '?', '-', '*', '\'', '`', '_', '\"', '(', ')', '{', '}', '[', ']', '<', '>',
          '¨', '"', '||', '/', '&', '~', '$', '\\', '#', '%']

num_re = r"^\d+$"
perc_re = r"^\d+/\d+$"
perc_re2 = r"^\d+/\d+-.*"
US_D_re = r"(\w+\$)(\d+.*)"
mm_dis_re = r"^(\d+)(mm)$"
# 30%-30% 分解

# 去除一个词两边的特殊符号, 当然不去这些符号和去两种训练词向量的方案都要尝试，边缘特征
def rm_edge_s(word):
    word = word.lower().strip()
    first, last = word[0], word[-1]
    while first in s_list:
        word = word[1:]
        if len(word) > 0:
            first = word[0]
        else:
            break
    while last in s_list:
        word = word[:-1]
        if len(word) > 0:
            last = word[-1]
        else:
            break
    return word


def get_sent_words_syns(sents):
    """
    index 记录当前EDU开始部分在句子中的位置
    :param sents:
    :return: edu在句子中的开始和结束
    """
    sents = sents.strip()
    temp_edu_tokens = None
    for token in sents.split():
        word_with_syn = get_word_syns(token)
        if temp_edu_tokens is None:
            temp_edu_tokens = word_with_syn
        else:
            temp_edu_tokens.extend(word_with_syn)

    # 省略号处理, 数字连续 处理
    final_edu_tokens = []
    count_end_pun = 0
    num_flag = False
    concat_num = False
    for token in temp_edu_tokens:
        if token == ".":
            count_end_pun += 1
        else:
            count_end_pun = 0
        if len(re.findall(num_re, token)) > 0:
            # 匹配到数字
            num_flag = True
        elif (len(re.findall(perc_re, token)) > 0 or len(re.findall(perc_re2, token)) > 0) and num_flag:
            # 匹配到百分比并且上一个是数字 立下 flag 当前操作要拼接
            concat_num = True
            num_flag = False
        else:
            # 没匹配到数字 或者 没匹配到百分比 直接num_flag设置为false
            num_flag = False

        final_edu_tokens.append(token)

        if count_end_pun == 3:
            final_edu_tokens.pop()
            final_edu_tokens.pop()
            final_edu_tokens.pop()
            final_edu_tokens.append("...")

        if concat_num:
            num_str2 = final_edu_tokens.pop()
            num_str1 = final_edu_tokens.pop()
            final_edu_tokens.append(num_str1 + num_str2)
            concat_num = False

        # US$120 --> US$ 120
        dollar_obj = re.search(US_D_re, token)
        if dollar_obj is not None:
            final_edu_tokens.pop()
            final_edu_tokens.append(dollar_obj.group(1))
            final_edu_tokens.append(dollar_obj.group(2))

        # 12mm --> 12 mm
        dist_obj = re.search(mm_dis_re, token)
        if dist_obj is not None:
            final_edu_tokens.pop()
            final_edu_tokens.append(dist_obj.group(1))
            final_edu_tokens.append(dist_obj.group(2))

        # 30%-owned or 30.1%-owned
        conn_re = r"^([\d.]+)(%)(-)([\w-]+)$"
        tmp_obj = re.search(conn_re, token)
        if tmp_obj is not None:
            final_edu_tokens.pop()
            final_edu_tokens.append(tmp_obj.group(1))
            final_edu_tokens.append(tmp_obj.group(2))
            final_edu_tokens.append(tmp_obj.group(3))
            final_edu_tokens.append(tmp_obj.group(4))

        # 1,20-$1,35 --> 120 - $ 135
        conn_re = r"^([\d,\-\w]+)(-)(\$)([\d,]+)$"
        tmp_obj = re.search(conn_re, token)
        if tmp_obj is not None:
            final_edu_tokens.pop()
            final_edu_tokens.append(tmp_obj.group(1))
            final_edu_tokens.append(tmp_obj.group(2))
            final_edu_tokens.append(tmp_obj.group(3))
            final_edu_tokens.append(tmp_obj.group(4))

        # Dsfs.,jhon --> Dsfs. , jhon
        conn_re = r"^(\w+\.)(,)(\w+)$"
        tmp_obj = re.search(conn_re, token)
        if tmp_obj is not None:
            final_edu_tokens.pop()
            final_edu_tokens.append(tmp_obj.group(1))
            final_edu_tokens.append(tmp_obj.group(2))
            final_edu_tokens.append(tmp_obj.group(3))

        # 1234=1213
        conn_re = r"^(\d+)(=)(\d+)$"
        tmp_obj = re.search(conn_re, token)
        if tmp_obj is not None:
            final_edu_tokens.pop()
            final_edu_tokens.append(tmp_obj.group(1))
            final_edu_tokens.append(tmp_obj.group(2))
            final_edu_tokens.append(tmp_obj.group(3))

        if token == "gonna":
            final_edu_tokens.pop()
            final_edu_tokens.append("gon")
            final_edu_tokens.append("na")

    return final_edu_tokens


def get_word_syns(token):
    """
    对一个token周围的符号切割得到保函符号和当前word的列表
    对 's的切割 以 's为单位
    :param token:
    :return:
    """
    token_list = []
    temp_list = []
    char_front_idx = 0
    char_last_idx = -1
    while char_front_idx < len(token) and (token[char_front_idx] in s_list):
        token_list.append(token[char_front_idx])
        char_front_idx += 1
    while char_last_idx >= -len(token) and (token[char_last_idx] in s_list) and (char_front_idx < len(token)):
        temp_list.insert(0, token[char_last_idx])
        char_last_idx -= 1
    if char_last_idx == -1:
        word = token[char_front_idx:]
        if len(word) > 0:
            token_list.append(word)
    else:
        word = token[char_front_idx:char_last_idx + 1]
        token_list.append(word)
        while char_last_idx + 1 <= -1:
            token_list.append(token[char_last_idx + 1])
            char_last_idx += 1
    token_list_final = []
    for token_ in token_list:
        if token_.endswith("'s") or token_.endswith("'m") or token_.endswith("'S") or token_.endswith("'d"):
            token_list_final.append(token_[:-2])
            token_list_final.append(token_[-2:])
        elif token_.endswith("'t") or token_.endswith("'re") or token_.endswith("'ll") or token_.endswith("'ve"):
            token_list_final.append(token_[:-3])
            token_list_final.append(token_[-3:])
        elif token_ == "cannot":
            token_list_final.append("can")
            token_list_final.append("not")
        else:
            token_list_final.append(token_)
    return token_list_final


if __name__ == "__main__":
    sent = "My test is 50.1%-owned"
    print(get_sent_words_syns(sent))
