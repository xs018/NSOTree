import math
import numpy as np
from lifelines.statistics import logrank_test


class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None
        self.log_p_value = None


def _build_binary_tree(values, t, e):
    if not values:
        return None

    # Create the root node
    root = Node(values[0])
    queue = [root]
    i = 1

    num_nodes = 2 ** len(values) - 1

    # Process each node, node's children are at indices 2n + 1, 2n + 2
    while i < num_nodes:
        # Get the current node from the front of the queue
        current = queue.pop(0)

        depth = int(math.log2(i+1))

        # Check if there is a left child; if yes, create it, link it, and add to the queue
        if i < num_nodes and values[depth] is not None:
            current.left = Node((1-values[depth])*current.value)
            queue.append(current.left)
        i += 1
        
        # Check if there is a right child; if yes, create it, link it, and add to the queue
        if i < num_nodes and values[depth] is not None:
            current.right = Node(values[depth]*current.value)
            queue.append(current.right)
        i += 1

        t_left, e_left = t[current.left.value], e[current.left.value]
        t_right, e_right = t[current.right.value], e[current.right.value]
        results = logrank_test(t_left, t_right, event_observed_A=e_left, event_observed_B=e_right)

        current.log_p_value = results.p_value

    return root

def _level_order_traversal(root):
    from collections import deque

    if not root:
        return []
    
    queue = deque([root])
    result = []
    
    while queue:
        node = queue.popleft()
        if node.log_p_value is not None:
            result.append(node.log_p_value)
        
        if node.left:
            queue.append(node.left)
        if node.right:
            queue.append(node.right)

    return result

def compute_logrank_test(masks, t, e):
    tree = _build_binary_tree(masks, t, e)
    return _level_order_traversal(tree)


if __name__ == "__main__":
    num_samples = 100
    m1 = np.random.randint(0, 2, (num_samples, ))
    m2 = np.random.randint(0, 2, (num_samples, ))
    m3 = np.random.randint(0, 2, (num_samples, ))

    t = np.random.randn(num_samples)
    e = np.random.randint(0, 2, (num_samples, ))

    d = 5
    tree = _build_binary_tree([np.random.randint(0, 2, (num_samples, )) for _ in range(d)], t, e)
    print(_level_order_traversal(tree))