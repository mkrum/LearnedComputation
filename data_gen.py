import random

class Node:
    def __init__(self, val, is_num = False, left = None, right = None):
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
    return eval(''.join(exp))


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
    n = random.random() * 1e7 * random.choice([-1, 1])
    if random.random() < 0.5:
        return int(n)
    return n


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
        x_train.append(split_exp)
        y_train.append(result)
    return x_train, y_train


if __name__ == "__main__":
    root = generate_random_infix_tree(['+', '*', '-', '/'], 5)
    exp = []
    exp = collapse_infix_tree(root)
    print(exp)
    print(split_expression_into_tokens(exp))
    print(eval_expression(exp))
    print(generate_training_examples(2, ['+', '*', '-', '/'], 5))