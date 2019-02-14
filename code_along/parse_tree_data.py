from pythonds.basic.stack import Stack
from pythonds.trees.binaryTree import BinaryTree

def make_tree(data):
    if data is None:
        return None
    tree = BinaryTree('')
    parse_chars = set(['(',')'])
    stack = Stack()
    stack.push(tree)
    currentTree = tree
    i = 0
    while i < len(data):
        d = data[i+1:]  # strip off first char, should always be '(' or ')'
        ci = data[i]  # ci is first char, either '(',')'
        next_index = next((j for j, ch in enumerate(d) if ch in parse_chars), None)
        if next_index == None:
            break
        curr_tuple = d[0:next_index].split()
        cf = d[next_index]  # cf is either '(',')'
        # sanity check
        assert ci in parse_chars
        assert cf in parse_chars

        if ci == '(':
            currentTree.setRootVal(curr_tuple)
            if cf == '(':
                stack.push(currentTree)
                currentTree.insertLeft('')
                currentTree = currentTree.getLeftChild()
            elif cf == ')':
                pass
        elif ci == ')':
            if cf == '(':
                parent = stack.pop()
                parent.insertRight('')
                currentTree = parent.getRightChild()
                stack.push(parent)
            elif cf == ')':
                parent = stack.pop()
                currentTree = parent
        i += next_index + 1     # extra +1 here because I always strip off leading '(' or ')'

    return tree


if __name__ == '__main__':
