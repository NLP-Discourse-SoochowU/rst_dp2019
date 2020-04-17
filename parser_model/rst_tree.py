import re
import copy
from config import USE_ELMo

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
        self.type = type_
        self.left_child = l_ch
        self.right_child = r_ch
        self.parent_node = p_node
        self.rel = rel
        self.child_rel = child_rel
        self.child_NS_rel = ch_ns_rel

        self.lines_list = lines_list
        self.temp_line = temp_line
        self.temp_edu = temp_edu
        self.temp_edu_span = temp_edu_span
        self.temp_edu_ids = temp_edu_ids
        self.temp_edu_emlo_emb = tmp_edu_emlo_emb
        self.temp_edus_count = temp_edus_count
        self.temp_edu_freq = temp_edu_freq
        self.temp_pos_ids = temp_pos_ids

        self.temp_edu_conn_ids = tmp_conn_ids

        self.temp_edu_heads = temp_edu_heads
        # False or True
        self.temp_edu_has_center_word = temp_edu_has_center_word

        self.edu_node = []

        self.rel_raw2coarse = rel_raw2coarse
        self.raw_rel = raw_rel

        self.feature_label = None
        self.inner_sent = False
        self.edu_node_boundary = False
        self.right_branch = right_branch

        self.attn_assigned = attn_assigned

    def __copy__(self):
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
        if temp_line_num > len(self.lines_list):
            return
        line = self.lines_list[temp_line_num]
        count_blank = get_blank(line)
        child_list = []
        while temp_line_num < len(self.lines_list):
            line = self.lines_list[temp_line_num]
            if get_blank(line) == count_blank:
                temp_line_num += 1
                node_type = type_re.findall(line)[0]
                if self.raw_rel:
                    node_new = rst_tree(type_=node_type, temp_line=line, rel=rel_re.findall(line)[0],
                                        lines_list=self.lines_list, raw_rel=self.raw_rel)
                else:
                    node_new = rst_tree(type_=node_type, temp_line=line, raw_rel=self.raw_rel,
                                        rel=self.rel_raw2coarse[rel_re.findall(line)[0]],
                                        lines_list=self.lines_list, rel_raw2coarse=self.rel_raw2coarse)
                if node_re.match(line):
                    temp_line_num = node_new.create_tree(temp_line_num=temp_line_num, p_node_=node_new)
                elif leaf_re.match(line):
                    pass
                child_list.append(node_new)
            else:
                while temp_line_num < len(self.lines_list) and end_re.match(self.lines_list[temp_line_num]):
                    temp_line_num += 1
                break
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
            temp_r.parent_node = new_node
            temp_l.parent_node = new_node
            child_list.append(new_node)

        self.right_child = child_list.pop()
        self.left_child = child_list.pop()
        self.right_child.parent_node = p_node_
        self.left_child.parent_node = p_node_
        if not self.right_child.rel == "span":
            self.child_rel = self.right_child.rel
        if not self.left_child.rel == "span":
            self.child_rel = self.left_child.rel
        self.child_NS_rel += self.left_child.type[0] + self.right_child.type[0]
        return temp_line_num

    def config_edus(self, temp_node, temp_edu_list, temp_edu_span_list, temp_eduids_list, edus_tag_ids_list,
                    edu_node, edus_conns_list, edu_headword_ids_list=None, edu_has_center_word_list=None,
                    total_edus_num=None, edus_boundary_list=None, temp_edu_freq_list=None, edu_emlo_emb_list=None):
        if self.right_child is not None and self.left_child is not None:
            self.left_child.config_edus(self.left_child, temp_edu_list, temp_edu_span_list, temp_eduids_list,
                                        edus_tag_ids_list, edu_node, edus_conns_list, edu_headword_ids_list,
                                        edu_has_center_word_list, total_edus_num, edus_boundary_list,
                                        temp_edu_freq_list, edu_emlo_emb_list)
            self.right_child.config_edus(self.right_child, temp_edu_list, temp_edu_span_list, temp_eduids_list,
                                         edus_tag_ids_list, edu_node, edus_conns_list, edu_headword_ids_list,
                                         edu_has_center_word_list, total_edus_num, edus_boundary_list,
                                         temp_edu_freq_list, edu_emlo_emb_list)
            self.temp_edu = self.left_child.temp_edu + " " + self.right_child.temp_edu
            self.temp_edu_span = (self.left_child.temp_edu_span[0], self.right_child.temp_edu_span[1])
            self.temp_edu_ids = self.left_child.temp_edu_ids + self.right_child.temp_edu_ids
            self.temp_pos_ids = self.left_child.temp_pos_ids + self.right_child.temp_pos_ids
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
            self.temp_edus_count = self.left_child.temp_edus_count + self.right_child.temp_edus_count

        elif self.right_child is None and self.left_child is None:
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
            self.temp_edus_count = 1
            self.temp_edu_heads = edu_headword_ids_list.pop(0)
            self.temp_edu_has_center_word = edu_has_center_word_list.pop(0)
            edu_node.append(temp_node)

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
