import random
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from lc.rep import ExpressionRep, BinaryOutputRep
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import pad_sequence
from bitstring import BitArray


class Node:
    def __init__(self, val, is_num=False, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        self.is_num = is_num


def eval_expression(exp):
    exp = [str(e) for e in exp]

    # Not safe but should be fine in this context.
    return eval("".join(exp))


def _collapse_infix_tree(root, exp):
    if not root:
        return

    _collapse_infix_tree(root.left, exp)
    exp.append(root.val)
    _collapse_infix_tree(root.right, exp)


def collapse_infix_tree(root):
    if not root:
        return []

    exp = []
    _collapse_infix_tree(root, exp)
    return exp


def get_random_num():
    n = random.random() * 10
    return int(n)


def fill_leaves_as_numbers(root):
    if not root or root.is_num:
        return

    if not root.left:
        root.left = Node(get_random_num(), is_num=True)
        fill_leaves_as_numbers(root.right)
    if not root.right:
        root.right = Node(get_random_num(), is_num=True)
        fill_leaves_as_numbers(root.left)

    fill_leaves_as_numbers(root.left)
    fill_leaves_as_numbers(root.right)


def generate_random_infix_tree(operators, nnodes):
    root = None
    valid_nodes = []
    while operators and nnodes > 0:
        random_op = random.choice(operators)
        nnodes -= 1
        if not valid_nodes:
            root = Node(random_op)
            valid_nodes.append(root)
            continue

        leaf_node = random.choice(valid_nodes)
        valid_nodes.remove(leaf_node)
        if leaf_node.left and leaf_node.right:
            continue

        new_leaf = Node(random_op)
        valid_nodes.append(new_leaf)
        if leaf_node.left:
            leaf_node.right = new_leaf
        elif leaf_node.right:
            leaf_node.left = new_leaf
        elif random.random() <= 0.5:
            leaf_node.left = new_leaf
            valid_nodes.append(leaf_node)
        else:
            leaf_node.right = new_leaf
            valid_nodes.append(leaf_node)

    # fill random numbers as leaves
    fill_leaves_as_numbers(root)
    return root


def fast_gen_no_tree_loop(num_examples, operators, num_operators, number_range):
    x_train = []
    y_train = []
    for _ in range(num_examples):
        x, y = fast_gen_no_tree(operators, num_operators, number_range)
        x_train.append(x)
        y_train.append(y)
    return x_train, y_train


def fast_gen_no_tree(operators, num_operators, number_range):

    low, high = number_range

    # High is exclusive, this will generate low to high inclusive
    first_num = np.random.randint(low=low, high=high + 1)

    exp = [first_num]
    while num_operators > 0:
        op = random.choice(operators)
        exp.append(op)
        num_operators -= 1
        num = np.random.randint(low=low, high=high + 1)
        exp.append(num)
    str_exp = [str(a) for a in exp]
    # result = eval(''.join(str_exp))

    return str_exp, str(eval_expression(str_exp))


class MathDataset(Dataset):
    def __init__(self, input_rep, output_rep, N=int(1e6), num_range=(-128, 127)):
        self.x, self.y = fast_gen_no_tree_loop(N, ["+", "-"], 1, num_range)
        self.input_rep = input_rep
        self.output_rep = output_rep

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        raw_x = self.x[idx]
        raw_y = self.y[idx]
        x_tensor = self.input_rep.from_str_list(raw_x).to_tensor()
        y_tensor = self.output_rep.from_str(raw_y).to_tensor()
        return x_tensor, y_tensor

    @classmethod
    def collate_fn(cls, batch):
        data, labels = zip(*batch)
        padded_data = pad_sequence(data, batch_first=True, padding_value=-1)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)
        return padded_data, padded_labels


class TestMathDataset(MathDataset):
    def __getitem__(self, idx):
        raw_x = self.x[idx]
        raw_y = self.y[idx]
        x_tensor = self.input_rep.from_str_list(raw_x).to_tensor()
        y_tensor = self.output_rep.from_str(raw_y).to_tensor()
        return x_tensor, y_tensor, raw_x, raw_y

    @classmethod
    def collate_fn(cls, batch):
        data, labels, raw_data, raw_labels = zip(*batch)
        padded_data = pad_sequence(data, batch_first=True, padding_value=-1)
        padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)
        return padded_data, padded_labels, raw_data, raw_labels

class FullTestMathDataset(TestMathDataset):

    def __init__(self, input_rep, output_rep, N=int(1e6), num_range=(-128, 127)):
        ops = ["+", "-"]
        self.x = []
        self.y = []
        for o in ops:
            for i in range(num_range[0], num_range[1] + 1):
                for j in range(num_range[0], num_range[1] + 1):
                    self.x.append([str(i), o, str(j)])
                    if o == "+":
                        self.y.append(str(i + j))
                    elif o == "-":
                        self.y.append(str(i - j))

        self.input_rep = input_rep
        self.output_rep = output_rep

if __name__ == "__main__":
    root = generate_random_infix_tree(["+", "*", "-", "/"], 5)
    exp = []
    exp = collapse_infix_tree(root)
    print(fast_gen_no_tree(["+"], 1))
    print(fast_gen_no_tree_loop(10, ["+"], 1))
