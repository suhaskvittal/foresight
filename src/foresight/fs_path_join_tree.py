"""
    author: Suhas Vittal
    date:   20 October 2021
"""

from fs_util import _path_to_swap_collection, _is_pow2

from copy import copy

class PathJoinTreeNode:
    def __init__(self, swap_collection, parent, left_child, right_child, target_index_list):
        self.data = swap_collection     # path(left) JOIN path(right)
        self.parent = parent
        self.left_child = left_child
        self.right_child = right_child
        # Data signals.
        self.dirty = 0      # 1 = dirty
        self.valid = True
        # Other data held by a node.
        self.target_index_list = target_index_list
        self.score = 0
        self.layout = None

class PathJoinTree:
    def __init__(self, leaves, verifier, kernel, future_gates): 
        self.kernel = kernel
        self.verifier = verifier
        self.future_gates = future_gates
        self.leaves = []
        for (i, path) in enumerate(leaves):
            soln = _path_to_swap_collection(path)
            pjnode = PathJoinTreeNode(soln, None, None, None, [i])
            _, _, pjnode.layout, _ = self.verifier._verify_and_measure(
                self.kernel,
                soln, 
                self.future_gates,
                verify_only=True
            ) 
            self.leaves.append(pjnode)
        while not _is_pow2(len(self.leaves)):
            self.leaves.append(PathJoinTreeNode([], None, None, None, []))
        # Build segmented tree from leaves using BFS-like algorithm.
        queue = [leaf for leaf in self.leaves]
        while len(queue) > 1:  # If len(queue) == 1, we only have the root inside.
            left, right = queue.pop(0), queue.pop(0)
            parent = PathJoinTreeNode(None, None, left, right, None)
            self._update_node(parent)
            left.parent = parent
            right.parent = parent
            queue.append(parent)  # Build parent by joining children.
        self.root = queue[0]  # candidate solution
    
    def get_root(self):
        return self.root.data

    def modify_leaves(self, modified_indices, new_paths):
        queue = []
        for (i, k) in enumerate(modified_indices):
            self.leaves[k].data = _path_to_swap_collection(new_paths[i])
            if self.leaves[k].parent is None:  # this is the root
                continue
            self.leaves[k].parent.dirty = 1
            queue.append(self.leaves[k].parent)
        while queue:
            node = queue.pop(0)
            if node.dirty == 0:  # already updated.
                continue
            # Else, perform update.
            self._update_node(node)
            node.dirty = 0
            # Place parent of node in the queue, mark it as dirty.
            if node.parent is not None:
                node.parent.dirty = 1
                queue.append(node.parent)
    
    def _update_node(self, node):
        left, right = node.left_child, node.right_child
        j_collection = []
        j_target_index_list = copy(left.target_index_list)
        
        j_target_index_list.extend(right.target_index_list)

        # Don't do any work if the join is not going to be valid anyways.
        if not (left.valid and right.valid):
            node.data = j_collection
            node.valid = False
            node.target_index_list = j_target_index_list
            return 

        # Join by inserting SWAPs from right.data whenever we can. 
        # Do not preemtively continue to other layers if we are not done 
        right_data_ptr = 0
        left_data_ptr = 0
        while right_data_ptr < len(right.data):
            right_layer = right.data[right_data_ptr]
            while right_layer:
                if left_data_ptr >= len(left.data):
                    j_collection.append(copy(right_layer))
                    break
                left_layer = left.data[left_data_ptr]

                joined_layer = []
                visited = set()
                for (v0, v1) in left_layer:
                    joined_layer.append((v0, v1))
                    visited.add(v0)
                    visited.add(v1)
                remaining_swaps = []    
                for (v0, v1) in right_layer:
                    if (v0, v1) in joined_layer or (v1, v0) in joined_layer:
                        continue
                    elif v0 in visited or v1 in visited:
                        remaining_swaps.append((v0, v1))
                    else:
                        joined_layer.append((v0, v1))
                    visited.add(v0)
                    visited.add(v1)
                j_collection.append(joined_layer)
                left_data_ptr += 1
                right_layer = remaining_swaps
            right_data_ptr += 1

        node.data = j_collection
        if node.parent is None:
            node.valid, node.score, node.layout, _ = self.verifier._verify_and_measure(
                self.kernel,
                j_collection, 
                self.future_gates,
                verify_only=False
            ) 
        else:
            node.valid, _, node.layout, _ = self.verifier._verify_and_measure(
                self.kernel,
                j_collection, 
                self.future_gates,
                verify_only=True
            ) 
        node.target_index_list = j_target_index_list

