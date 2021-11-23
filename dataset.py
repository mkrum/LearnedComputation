
import random
from torch.utils.data import DataLoader, Dataset
from rep import ExpressionRep

class Node:
    def __init__(self, val, is_num=False, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        self.is_num = is_num


def split_expression_into_tokens(exp):
    unpacked = []
    for e in exp:
        unpacked.extend(list(str(e)))
    return unpacked


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
    # if random.random() < 0.5:
    return int(n)
    # return n


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


def generate_training_examples(num_examples, operators, num_operator_nodes):
    x_train = []
    y_train = []
    for _ in range(num_examples):
        infix_root = generate_random_infix_tree(operators, num_operator_nodes)
        exp = collapse_infix_tree(infix_root)
        result = eval_expression(exp)
        split_exp = split_expression_into_tokens(exp)
        result = str(result)
        x_train.append(split_exp)
        y_train.append(result)
    return x_train, y_train


def fast_gen_no_tree_loop(num_examples, operators, num_operators):
    x_train = []
    y_train = []
    for _ in range(num_examples):
        x, y = fast_gen_no_tree(operators, num_operators)
        x_train.append(x)
        y_train.append(y)
    return x_train, y_train


def fast_gen_no_tree(operators, num_operators):
    first_num = int(random.random() * 10000)
    exp = [first_num]
    while num_operators > 0:
        op = random.choice(operators)
        exp.append(op)
        num_operators -= 1
        exp.append(int(random.random() * 10000))
    str_exp = [str(a) for a in exp]
    # result = eval(''.join(str_exp))

    return split_expression_into_tokens(str_exp), str(eval_expression(str_exp))

class SimpleExpressionDataset(Dataset):

    def __init__(self, N=int(1e6)):
        self.x, self.y = fast_gen_no_tree_loop(N, ['+', '-'], 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        raw_x = self.x[idx]
        raw_y = ['<start>'] + list(self.y[idx]) + ['<stop>']
        x_tensor = ExpressionRep.from_str_list(raw_x).to_tensor()
        y_tensor = ExpressionRep.from_str_list(raw_y).to_tensor()
        return (x_tensor, y_tensor)

if __name__ == "__main__":
    root = generate_random_infix_tree(["+", "*", "-", "/"], 5)
    exp = []
    exp = collapse_infix_tree(root)
    print(fast_gen_no_tree(["+"], 1))
    print(fast_gen_no_tree_loop(10, ["+"], 1))

