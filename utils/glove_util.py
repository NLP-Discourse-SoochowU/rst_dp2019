import numpy as np
from utils.file_util import *
from config import EMBED_SIZE, GLOVE_PATH
"""
    创建word2filename  word2vec的文件
"""
def build_glove_dict():
    """
    对指定维度的glove向量库创建数据字典
    :return:
    """
    dim = EMBED_SIZE
    save_dir = os.path.join(GLOVE_PATH, "gloves_"+str(dim))
    safe_mkdir(save_dir)
    file_tag = 0
    count_line = 0
    f_n = "part_"
    tmp_file = os.path.join(save_dir, f_n + str(file_tag) + ".pkl")
    tmp_dict = dict()
    word2file_dict = dict()
    with open(os.path.join(GLOVE_PATH, "glove.6B."+str(EMBED_SIZE)+"d.txt"), "r") as f:
        for line in f:
            count_line += 1
            line_ = line.strip().split()
            if len(line_) < 25:
                continue
            vec = np.array(line_[-dim:])
            word = ' '.join(line_[:-dim])
            tmp_dict[word] = vec.astype(np.float32)
            word2file_dict[word] = f_n + str(file_tag) + ".pkl"
            if count_line % 500000 == 0:  # 预计50万没问题
                with open(tmp_file, "wb") as f_:
                    pkl.dump(tmp_dict, f_)
                tmp_dict = dict()
                file_tag += 1
                tmp_file = os.path.join(save_dir, f_n + str(file_tag) + ".pkl")
        # 余数存储
        if len(tmp_dict.keys()) > 0:
            with open(tmp_file, "wb") as f_:
                pkl.dump(tmp_dict, f_)

    # 对文件名映射信息进行存储
    with open(os.path.join(save_dir, "word2filename.pkl"), "wb") as f:
        pkl.dump(word2file_dict, f)
