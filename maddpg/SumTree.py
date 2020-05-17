import numpy

"""
    The sum tree enables efficient querying of using random numbers. 
    By quoting the sum of all elements, a random number can be generated and this random number then returns an element
    based on the priority value. The bigger the priority value, the higher the sampler priority. 
    
"""


class SumTree:
    # Which index to write to
    write = 0

    def __init__(self, capacity):
        """
            preallocate memory for the tree because the size is already known
            the tree is flattened by taking the rows reading in from the top
        """
        # memory lengths
        self.capacity = capacity
        # 0 chance of randomly selecting empty cells
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        # the number of actual entries
        self.n_entries = 0

    # update to the root node
    # go recursively up the tree and add the amount the sums have changed
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        # recursion done if top (node 0) is reached.
        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    # traverse the tree from top to bottom
    def _retrieve(self, idx, s):
        # left child
        left = 2 * idx + 1
        # right child
        right = left + 1

        # check if the bottom of the tree has been reached
        if left >= len(self.tree):
            return idx

        # go left if value fits in left tree value
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        # go right, subtract the value of the left tree node
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        # the total sum in the tree
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update tree value calling propagate for recursive upwards update
    def update(self, idx, p):
        # just add the change
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])
