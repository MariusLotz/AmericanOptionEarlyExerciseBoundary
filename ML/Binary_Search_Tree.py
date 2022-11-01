class Node:
    def __init__(self, rqsigma, border_vec):
        self.left = None
        self.right = None
        self.rqs = rqsigma
        self.data = border_vec


class Binary_Search_Tree(Node):
    def __init__(self):
        self.root = None

    def create_from_sample(self, txt_file):
        with open(txt_file) as file:
            lines = file.readlines()
        for line in lines:
            d = []
            d1 = line[]
            d.append()

    def create_search_tree(self, data):
        for d in data:
            node = Node(d[0], d[1])
            self.add_Node(node)

    def is_greater(self, greater_x, x):
        if greater_x[0] < x[0]: return False
        elif greater_x[0] == x[0]:
            if greater_x[1] < x[1]: return False
            elif greater_x[1] == x[1]:
                if greater_x[2] < x[2]: return False
        else: return True

    def add_Node(self, node):
        if self.root is None:
            self.root = node
        else:
            self._add_Node(node)

    def _add_Node(self, at_node, new_node):
        if self.is_greater(at_node.rqs, new_node.rqs):
            if at_node.left is not None:
                self._add_Node(at_node.left, new_node)
            else:
                at_node.left = new_node
        else:
            if at_node.right is not None:
                self._add_Node(at_node.right, new_node)
            else:
                at_node.right = new_node

    def getRoot(self):
        return self.root

    def deleteTree(self):
        # garbage collector will do this for us.
        self.root = None

    def find(self, val):
        if self.root is not None:
            return self._find(val, self.root)
        else:
            return None

    def _find(self, val, node):
        if val == node.v:
            return node
        elif (val < node.v and node.l is not None):
            return self._find(val, node.l)
        elif (val > node.v and node.r is not None):
            return self._find(val, node.r)