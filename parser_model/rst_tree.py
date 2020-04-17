"""
建立一个可以操作的单独的rst树类
    给左右孩子 当前节点
    给根据文档wsj的dis文件建立树的方法，返回根root
    ( Root (span 1 3)
      ( Nucleus (leaf 1) (rel2par span) (text _!Spencer J. Volk, president and chief operating officer of this consumer and industrial products company, was elected a director._!) )
      ( Satellite (span 2 3) (rel2par elaboration-additional)
        ( Nucleus (leaf 2) (rel2par span) (text _!Mr. Volk, 55 years old, succeeds Duncan Dwight,_!) )
        ( Satellite (leaf 3) (rel2par elaboration-additional-e) (text _!who retired in September._!) )
      )
    )
    运用前序遍历的方法就可以创建，只要遍历过程中发现是span就作为node递归下去，标记行数
    
    问题： 关系抽取中遇到关系怎么是结构关系，因为之前DPLP训练出来的操作对应的关系是内容深层次的关系
单个对象就是tree node 
    作者： 张龙印
    日期： 2018.3.11
"""
import re
import copy
from config import USE_ELMo

# 定义模式匹配
leaf_parttern = r' *\( \w+ \(leaf .+'
leaf_re = re.compile(leaf_parttern)
node_parttern = r' *\( \w+ \(span .+'
node_re = re.compile(node_parttern)
end_parttern = r'\s*\)\s*'
end_re = re.compile(end_parttern)
nodetype_parttern = r' *\( (\w+) .+'
type_re = re.compile(nodetype_parttern)
rel_parttern = r' *\( \w+ \(.+\) \(rel2par ([\w-]+).+'
rel_re = re.compile(rel_parttern)
node_leaf_parttern = r' *\( \w+ \((\w+) \d+.*\).+'
node_leaf_re = re.compile(node_leaf_parttern)


def get_blank(line):
    count = 0
    while line[count] == " ":
        count += 1
    return count


class rst_tree:
    def __init__(self, type_=None, l_ch=None, r_ch=None, p_node=None, child_rel=None, rel=None, ch_ns_rel="",
                 lines_list=None, temp_line=" ", file_name=None, rel_raw2coarse=None, temp_edu=None, temp_edu_span=None,
                 temp_edu_ids=None, temp_pos_ids=None, tmp_conn_ids=None, temp_edu_heads=None,
                 temp_edu_has_center_word=None, raw_rel=False, temp_edu_freq=None, temp_edus_count=0,
                 tmp_edu_emlo_emb=None, right_branch=False, attn_assigned=None):
        self.file_name = file_name
        # 类型为三种 Satellite Nucleus Root
        self.type = type_
        self.left_child = l_ch
        self.right_child = r_ch
        self.parent_node = p_node
        # 当前结点上标注的rel
        self.rel = rel
        self.child_rel = child_rel
        self.child_NS_rel = ch_ns_rel

        # 动态获取的文档对象
        self.lines_list = lines_list
        # temp_line当前节点在文件中的描述
        self.temp_line = temp_line
        # 当前span的拼接结果
        self.temp_edu = temp_edu
        self.temp_edu_span = temp_edu_span
        self.temp_edu_ids = temp_edu_ids
        self.temp_edu_emlo_emb = tmp_edu_emlo_emb  # 这个只有在EMLo的方案下使用
        self.temp_edus_count = temp_edus_count
        self.temp_edu_freq = temp_edu_freq
        self.temp_pos_ids = temp_pos_ids  # 后面需要，所以针对EDU级别的pos_ids进行填充，注意这里是对句子进行词性分析

        self.temp_edu_conn_ids = tmp_conn_ids

        # 当前EDU前三个词的head word 对应的ids
        self.temp_edu_heads = temp_edu_heads
        # False or True
        self.temp_edu_has_center_word = temp_edu_has_center_word

        self.edu_node = []

        self.rel_raw2coarse = rel_raw2coarse
        self.raw_rel = raw_rel

        self.feature_label = None
        self.inner_sent = False  # 当前节点是否包含在一个句子内部
        self.edu_node_boundary = False  # 当前节点是否是句子右边界节点
        self.right_branch = right_branch  # 用来标记执行了right_branch得到的左孩子的叶子结点

        self.attn_assigned = attn_assigned  # 为各个内部节点创建attn的分布属性，在构建过程中进行attn的获取

    def __copy__(self):
        """
        对于tree_obj的拷贝涉及对于EDU拷贝，即为当前类
        :return:
        """
        new_obj = rst_tree(type_=self.type, l_ch=copy.copy(self.left_child), r_ch=copy.copy(self.right_child),
                           p_node=copy.copy(self.parent_node), child_rel=self.child_rel, rel=self.rel,
                           ch_ns_rel=self.child_NS_rel, lines_list=self.lines_list[:], temp_line=self.temp_line,
                           file_name=self.file_name, rel_raw2coarse=None, tmp_conn_ids=None, raw_rel=self.raw_rel)
        return new_obj

    def append(self, root):
        pass

    def get_type(self, line):
        pass

    def create_tree(self, temp_line_num, p_node_=None):
        """
        以当前行数为起点的一个节点树的构建，这样保证了递归实现的可行性
        构建了各个节点的主次类别，关系类别，孩子关系类别，孩子主次关系，父母节点指向，当前行的内容
        张龙印 2017年12月15日
        :param temp_line_num:
        :param p_node_:
        :return:
        """
        if temp_line_num > len(self.lines_list):
            return
        line = self.lines_list[temp_line_num]
        # 为了判断当前line的父节点是否存在第三个孩子，我们在第一个孩子这里取空串长度作为哨
        count_blank = get_blank(line)
        child_list = []  # 用于存储当前所有的孩子节点
        while temp_line_num < len(self.lines_list):
            line = self.lines_list[temp_line_num]
            if get_blank(line) == count_blank:  # 下面一行还是孩子节点
                temp_line_num += 1
                node_type = type_re.findall(line)[0]
                if self.raw_rel:
                    node_new = rst_tree(type_=node_type, temp_line=line, rel=rel_re.findall(line)[0],
                                        lines_list=self.lines_list, raw_rel=self.raw_rel)
                else:
                    node_new = rst_tree(type_=node_type, temp_line=line, raw_rel=self.raw_rel,
                                        rel=self.rel_raw2coarse[rel_re.findall(line)[0]],
                                        lines_list=self.lines_list, rel_raw2coarse=self.rel_raw2coarse)
                # input(rel_re.findall(line)[0] + "=====" + node_new.rel)
                # 是节点就继续
                if node_re.match(line):
                    temp_line_num = node_new.create_tree(temp_line_num=temp_line_num, p_node_=node_new)
                elif leaf_re.match(line):
                    pass
                child_list.append(node_new)
            else:
                # 若下一行不是孩子节点, 则对最右端孩子之后进行行递增
                while temp_line_num < len(self.lines_list) and end_re.match(self.lines_list[temp_line_num]):
                    temp_line_num += 1
                break
        # 对当前root_list[]里面的所有孩子节点和p_root关联起来
        while len(child_list) > 2:
            temp_r = child_list.pop()
            temp_l = child_list.pop()
            if not temp_l.rel == "span":
                new_node = rst_tree(type_="Nucleus", r_ch=temp_r, l_ch=temp_l, ch_ns_rel="NN", child_rel=temp_l.rel,
                                    rel=temp_l.rel, temp_line="<new created line>", lines_list=self.lines_list,
                                    raw_rel=self.raw_rel)
            if not temp_r.rel == "span":
                new_node = rst_tree(type_="Nucleus", r_ch=temp_r, l_ch=temp_l, ch_ns_rel="NN", child_rel=temp_r.rel,
                                    rel=temp_r.rel, temp_line="<new created line>", lines_list=self.lines_list,
                                    raw_rel=self.raw_rel)
            new_node.temp_line = "<new created line>"
            # 指向父节点
            temp_r.parent_node = new_node
            temp_l.parent_node = new_node
            # 创建结束，将新结点入栈
            child_list.append(new_node)

        self.right_child = child_list.pop()
        self.left_child = child_list.pop()
        # 指向父节点
        self.right_child.parent_node = p_node_
        self.left_child.parent_node = p_node_
        # 孩子rel关系
        if not self.right_child.rel == "span":
            self.child_rel = self.right_child.rel
        if not self.left_child.rel == "span":
            self.child_rel = self.left_child.rel
        # 孩子NS关系, 只取第一个字母
        self.child_NS_rel += self.left_child.type[0] + self.right_child.type[0]
        return temp_line_num

    def config_edus(self, temp_node, temp_edu_list, temp_edu_span_list, temp_eduids_list, edus_tag_ids_list,
                    edu_node, edus_conns_list, edu_headword_ids_list=None, edu_has_center_word_list=None,
                    total_edus_num=None, edus_boundary_list=None, temp_edu_freq_list=None, edu_emlo_emb_list=None):
        """
        对建立好的树的再次优化，递归实现对各个span的EDU的拼接
        对于span的edu就用孩子的edu进行拼接
        对于span的edu_ids用孩子的edu被padding之后的结果进行拼接
        :param edu_emlo_emb_list:
        :param temp_edu_freq_list:
        :param total_edus_num:
        :param edu_has_center_word_list:
        :param edu_headword_ids_list:
        :param edus_conns_list:
        :param edus_tag_ids_list:
        :param temp_eduids_list: EMLo: edus_ids_list = (edu_num, word_num, id_len)
        :param temp_edu_span_list:
        :param edus_boundary_list:
        :param temp_edu_list:
        :param temp_node:
        :param edu_node:
        :return:
        """
        if self.right_child is not None and self.left_child is not None:
            self.left_child.config_edus(self.left_child, temp_edu_list, temp_edu_span_list, temp_eduids_list,
                                        edus_tag_ids_list, edu_node, edus_conns_list, edu_headword_ids_list,
                                        edu_has_center_word_list, total_edus_num, edus_boundary_list,
                                        temp_edu_freq_list, edu_emlo_emb_list)
            self.right_child.config_edus(self.right_child, temp_edu_list, temp_edu_span_list, temp_eduids_list,
                                         edus_tag_ids_list, edu_node, edus_conns_list, edu_headword_ids_list,
                                         edu_has_center_word_list, total_edus_num, edus_boundary_list,
                                         temp_edu_freq_list, edu_emlo_emb_list)
            # 当前节点不是叶子节点，edus取孩子节点的拼接结果
            self.temp_edu = self.left_child.temp_edu + " " + self.right_child.temp_edu
            self.temp_edu_span = (self.left_child.temp_edu_span[0], self.right_child.temp_edu_span[1])
            self.temp_edu_ids = self.left_child.temp_edu_ids + self.right_child.temp_edu_ids
            self.temp_pos_ids = self.left_child.temp_pos_ids + self.right_child.temp_pos_ids
            # 对父节点的边界和inner_sent判定
            if self.left_child.inner_sent is False or self.right_child.inner_sent is False:
                self.inner_sent = False
                self.edu_node_boundary = False
            elif self.left_child.edu_node_boundary is False:
                self.inner_sent = True
                if self.right_child.edu_node_boundary is True:
                    self.edu_node_boundary = True
                else:
                    self.edu_node_boundary = False
            else:
                self.inner_sent = False
                self.edu_node_boundary = False
            # 当前区域包含的EDU个数
            self.temp_edus_count = self.left_child.temp_edus_count + self.right_child.temp_edus_count

        elif self.right_child is None and self.left_child is None:
            # 叶节点
            self.temp_edu = temp_edu_list.pop(0)
            self.temp_edu_span = temp_edu_span_list.pop(0)
            self.temp_edu_ids = temp_eduids_list.pop(0)
            if USE_ELMo:
                self.temp_edu_emlo_emb = edu_emlo_emb_list.pop(0)
            self.temp_pos_ids = edus_tag_ids_list.pop(0)
            self.temp_edu_conn_ids = edus_conns_list.pop(0)
            self.edu_node_boundary = edus_boundary_list.pop(0)
            self.temp_edu_freq = temp_edu_freq_list.pop(0)
            self.inner_sent = True
            # 当前区域包含的edu个数
            self.temp_edus_count = 1
            # dependency
            self.temp_edu_heads = edu_headword_ids_list.pop(0)  # 当前EDU所有tokens对应的head word的ids
            self.temp_edu_has_center_word = edu_has_center_word_list.pop(0)  # 当前EDU是否包含句子中心词
            # input(self.temp_edu_conn_ids)
            edu_node.append(temp_node)

            # feature label form
            length_of_edu = len(self.temp_edu_ids)
            if length_of_edu <= 5:
                length_label = 1
            elif 15 >= length_of_edu > 5:
                length_label = 2
            else:
                length_label = 3

            edus_left = len(temp_eduids_list)
            if edus_left > total_edus_num*3/4:
                position_label = 1
            elif edus_left > total_edus_num/2:
                position_label = 2
            elif edus_left > total_edus_num/4:
                position_label = 3
            else:
                position_label = 4

            if self.temp_edu_has_center_word:
                self.feature_label = "T_l"+str(length_label)+"_p"+str(position_label)
            else:
                self.feature_label = "F_l" + str(length_label) + "_p" + str(position_label)

    def config_nodes(self, root):
        if root is None or root.left_child is None or root.right_child is None:
            return
        self.config_nodes(root.left_child)
        self.config_nodes(root.right_child)
        root.temp_edu = root.left_child.temp_edu + " " + root.right_child.temp_edu
        root.temp_edu_span = (root.left_child.temp_edu_span[0], root.right_child.temp_edu_span[1])
        root.temp_edu_ids = root.left_child.temp_edu_ids + root.right_child.temp_edu_ids
        root.temp_pos_ids = root.left_child.temp_pos_ids + root.right_child.temp_pos_ids

    def pre_traverse(self):
        """
        验证树建立的合法性，进行前序遍历
            1希望建立的树和文件保持一致
            2希望shift-reduce过程和configuration保持同步，shift-reduce过程和文件的
            后续遍历保持一致，验证。
        :return:
        """
        if self.right_child is not None and self.left_child is not None:
            self.left_child.pre_traverse()
            self.right_child.pre_traverse()
            print("child_rel: ", self.child_rel)
            print("edu: ", self.temp_edu)
            print(len(self.temp_edu.split()))
            print("edu_ids: ", self.temp_edu_ids)
            print(len(self.temp_edu_ids))
            print("POS_IDS: ", self.temp_pos_ids)
            print(self.temp_edu_span)
        else:
            print("edu: ", self.temp_edu)
            print(len(self.temp_edu.split()))
            print("edu_ids: ", self.temp_edu_ids)
            print(len(self.temp_edu_ids))
            print("POS_IDS: ", self.temp_pos_ids)
            print(self.temp_edu_span)
        input(self.rel)
        # if self.right_child is not None and self.left_child is not None:
        #     self.left_child.pre_traverse()
        #     self.right_child.pre_traverse()
        # print(self.temp_edu)
        # print("是否是边界：", self.edu_node_boundary)
        # print("当前节点是否在句子内部：", self.inner_sent)
        # input("inputting...")
