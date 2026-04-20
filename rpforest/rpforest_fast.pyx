#!python
# distutils: language = c++
# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import struct
from struct import pack, calcsize, unpack_from

import numpy as np
cimport numpy as cnp

from libc.stdlib cimport malloc, free

cdef extern from "string.h":
    void memcpy(void* des, void* src, int size)

from libcpp.algorithm cimport sort
from cython.operator cimport dereference as deref, preincrement as inc, predecrement as dec
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcpp.set cimport set as cset
from libcpp.unordered_set cimport unordered_set
from libcpp.queue cimport priority_queue
from libcpp.map cimport map as cmap


ctypedef float hyp
cdef str hyp_symbol = '@f'

cdef unsigned int uint_size = calcsize('@I')
cdef unsigned int int_size = calcsize('@i')
cdef unsigned int uchar_size = calcsize('@B')
cdef unsigned int hyp_size = calcsize(hyp_symbol)


cdef unsigned int SERIALIZATION_VERSION = 2



### The Ben zone

cdef void add_vectors_inplace(double[::1] vec1, double[::1] vec2, unsigned int dim) noexcept nogil:
    """
    Add vec2 to vec1 in-place.
    """
    cdef unsigned int i
    
    for i in range(dim):
        vec1[i] += vec2[i]

cdef inline double l2_norm(double[::1] vec, unsigned int dim) nogil:
    """
    Compute the L2 (Euclidean) norm of a vector.
    """
    cdef unsigned int i
    cdef double result = 0.0
    
    for i in range(dim):
        result += vec[i] * vec[i]
    
    return result ** 0.5

cdef inline void normalize_vector(double[::1] vec, unsigned int dim) noexcept nogil:
    """
    Normalize a vector in-place by its L2 norm.
    """
    cdef unsigned int i
    cdef double norm = l2_norm(vec, dim)
    
    if norm > 0.0:
        for i in range(dim):
            vec[i] /= norm

cdef inline void zero_vector(double[::1] vec, unsigned int dim) noexcept nogil:
    """
    Zero out a vector in-place.
    """
    cdef unsigned int i
    for i in range(dim):
        vec[i] = 0.0

cdef inline void subtract_vectors_inplace(double[::1] vec1, double[::1] vec2, unsigned int dim) noexcept nogil:
    """
    Subtract vec2 from vec1 in-place.
    """
    cdef unsigned int i
    for i in range(dim):
        vec1[i] -= vec2[i]

cdef void update_query_prime_nogil(double[::1] q_prime,
                                    double[:, ::1] X,
                                    unsigned int dim,
                                    unsigned int n,
                                    unsigned int warmup,
                                    unsigned int i,
                                    unordered_set[int] *added_this_iter,
                                    unordered_set[int] *removed_this_iter,
                                    cset[pair[double, int]] *internal_candidates) noexcept nogil:
    """
    Update q_prime vector efficiently with nogil.
    """
    global tree_roots, tree_hyperplanes, num_roots


    cdef int idx
    cdef cset[pair[double, int]].iterator it
    
    if i >= warmup:
        if deref(removed_this_iter).size() > (n / 2) or i == warmup:
            # Do full computation
            zero_vector(q_prime, dim)
            
            it = deref(internal_candidates).begin()
            while it != deref(internal_candidates).end():
                add_vectors_inplace(q_prime, X[deref(it).second, :], dim)
                inc(it)
        else:
            # Incrementally update q_prime
            for idx in deref(added_this_iter):
                add_vectors_inplace(q_prime, X[idx, :], dim)
            
            for idx in deref(removed_this_iter):
                subtract_vectors_inplace(q_prime, X[idx, :], dim)



# Helpers for the mtq tree

cdef Node **tree_roots = <Node**>NULL
cdef Hyperplanes **tree_hyperplanes = <Hyperplanes**>NULL
cdef Py_ssize_t num_roots = 0


cpdef void build_global_tree_ptrs(list trees):
    """
    Builds some list that MTQ needs.
    Used to be done at the top of MTQ.
    """
    global tree_roots, tree_hyperplanes, num_roots

    cdef Py_ssize_t i
    cdef Tree tree

    # free old
    if tree_roots != NULL:
        free(tree_roots)
        tree_roots = <Node**>NULL
    if tree_hyperplanes != NULL:
        free(tree_hyperplanes)
        tree_hyperplanes = <Hyperplanes**>NULL
    num_roots = 0

    num_roots = len(trees)
    if num_roots == 0:
        return

    tree_roots = <Node**>malloc(num_roots * sizeof(Node*))
    if tree_roots == NULL:
        raise MemoryError()

    tree_hyperplanes = <Hyperplanes**>malloc(num_roots * sizeof(Hyperplanes*))
    if tree_hyperplanes == NULL:
        free(tree_roots)
        tree_roots = <Node**>NULL
        raise MemoryError()

    for i in range(num_roots):
        tree = trees[i]
        tree_roots[i] = tree.root
        tree_hyperplanes[i] = tree.hyperplanes

cpdef void free_global_tree_ptrs():
    global tree_roots, tree_hyperplanes, num_roots

    if tree_roots != NULL:
        free(tree_roots)
        tree_roots = <Node**>NULL
    if tree_hyperplanes != NULL:
        free(tree_hyperplanes)
        tree_hyperplanes = <Hyperplanes**>NULL
    num_roots = 0


cpdef query_all_mtq(double[::1] x, double[:, ::1] X, list trees, unsigned int n, unsigned int warmup):
    """
    Return approximate nearest neighbours to point x from the model.
    Uses the MTQ technique.
    """
    global tree_roots, tree_hyperplanes, num_roots

    cdef unsigned int i
    cdef unsigned int dim, no_returns

    cdef cnp.ndarray[int, ndim=1] result
    cdef unordered_set[int] seen_ids

    cdef cset[pair[double, int]] internal_candidates
    cdef cnp.ndarray[double, ndim=1] q_prime = np.copy(x)
    cdef double[::1] q_prime_view = q_prime
    cdef cnp.ndarray[double, ndim=1] q_normalized = np.empty(x.shape[0], dtype=np.float64)
    cdef double[::1] q_normalized_view = q_normalized

    cdef unordered_set[int] removed_this_iter;
    cdef unordered_set[int] added_this_iter;
 
    dim = X.shape[1]

    # Entire loop now runs in nogil context
    with nogil:
        for i in range(num_roots):
            # Clear the sets
            added_this_iter.clear()
            removed_this_iter.clear()

            # Normalize q_prime for tree query and get candidates
            q_normalized_view[:] = q_prime_view
            normalize_vector(q_normalized_view, dim)
            _get_candidates_at_tree_i_nogil(q_normalized_view, x, X, tree_roots[i], tree_hyperplanes[i], dim, n, &seen_ids, &added_this_iter, &removed_this_iter, &internal_candidates)
            update_query_prime_nogil(q_prime_view, X, dim, n, warmup, i, &added_this_iter, &removed_this_iter, &internal_candidates)
    
    no_returns = min(n, internal_candidates.size())
    result = np.empty(no_returns, dtype=np.int32)

    i = 0
    for candidate in internal_candidates:
        if i >= no_returns:
            break
        result[i] = candidate.second
        i += 1

    return result

    

cdef void _get_candidates_at_tree_i_nogil(double[::1] q_prime,
                                double[::1] original_query,
                                double[:, ::1] X,
                                Node *root,
                                Hyperplanes *hyperplanes,
                                int dim,
                                unsigned int n,
                                unordered_set[int] *seen_ids,
                                unordered_set[int] *added_this_iter,
                                unordered_set[int] *removed_this_iter,
                                cset[pair[double, int]] *internal_candidates) noexcept nogil:
    """
    Get all members of x's leaf nodes.
    Accepts raw pointers for nogil compatibility.
    """

    cdef Node *leaf

    leaf = query(root, hyperplanes, q_prime)

    sort_candidates_mtq(original_query, X, dim, n, leaf.indices, seen_ids, added_this_iter, removed_this_iter, internal_candidates)

cdef void _get_candidates_at_tree_i(double[::1] q_prime,
                                double[::1] original_query,
                                double[:, ::1] X,
                                Tree tree,
                                int dim,
                                unsigned int n,
                                unordered_set[int] *seen_ids,
                                unordered_set[int] *added_this_iter,
                                unordered_set[int] *removed_this_iter,
                                cset[pair[double, int]] *internal_candidates) noexcept nogil:
    """
    Get all members of x's leaf nodes.
    Now accepts a Tree object directly and runs with nogil.
    """

    cdef Node *root
    cdef Node *leaf

    root = tree.root
    leaf = query(root, tree.hyperplanes, q_prime)

    sort_candidates_mtq(original_query, X, dim, n, leaf.indices, seen_ids, added_this_iter, removed_this_iter, internal_candidates)

cdef void sort_candidates_mtq(
                                double[::1] original_query,
                                double[:, ::1] X,
                                unsigned int dim,
                                unsigned int n,
                                vector[int] *candidates,
                                unordered_set[int] *seen_ids,
                                unordered_set[int] *added_this_iter,
                                unordered_set[int] *removed_this_iter,
                                cset[pair[double, int]] *internal_candidates) noexcept nogil:
    """
    Perform final cosine similarity sorting step on merged candidates.
    """
    cdef unsigned int i, j, no_candidates
    cdef int idx
    cdef double dst
    cdef cset[pair[double, int]].iterator last_it

    no_candidates = candidates.size()

    for i in range(no_candidates):
        idx = deref(candidates)[i]

        # If not in seen_ids
        if deref(seen_ids).find(idx) == deref(seen_ids).end():
            dst = 0.0
            for j in range(dim):
                dst -= original_query[j] * X[idx, j]

            # If this is smaller than our frutherst thing or we don't have enough points
            if deref(internal_candidates).size() < n or dst < deref(deref(internal_candidates).rbegin()).first:
                deref(internal_candidates).insert(pair[double, int](dst, idx))
                deref(added_this_iter).insert(idx)

                if deref(internal_candidates).size() > n:
                    # Get the last element (largest distance) - sets are ordered, so last element has max key
                    last_it = deref(internal_candidates).end()
                    dec(last_it)  # Decrement iterator to get last valid element

                    # Add to popped
                    deref(removed_this_iter).insert(deref(last_it).second)
                    
                    # Remove the last element
                    deref(internal_candidates).erase(last_it)

            deref(seen_ids).insert(idx)
    
### End Ben zone



cdef inline double dot(hyp *x, double[::1] y, unsigned int n) nogil:

    cdef unsigned int i
    cdef double result = 0.0

    for i in range(n):
        result += x[i] * y[i]

    return result


cpdef list encode_all(double[::1] x, list trees):
    """
    Return leaf codes for point x for all trees.
    """

    cdef unsigned int i, dim
    cdef Tree tree
    cdef list code
    dim = x.shape[0]

    cdef list codes = []

    for i in range(len(trees)):
        tree = trees[i]
        root = tree.root
        code = []
        encode(root, tree.hyperplanes, x, code, dim)
        codes.append('%s:' % i + ''.join(code))

    return codes


cdef vector[pair[double, int]] sort_candidates(double[::1] x,
                                               double[:, ::1] X,
                                               unsigned int dim,
                                               vector[int] candidates) noexcept nogil:
    """
    Perform final cosine similarity sorting step on merged candidates.
    """

    cdef vector[pair[double, int]] scores
    cdef unsigned int i, j, no_candidates
    cdef int idx, prev_idx
    cdef double dst

    no_candidates = candidates.size()
    scores.reserve(no_candidates)
    sort(candidates.begin(), candidates.end())

    prev_idx = -1
    for i in range(no_candidates):

        idx = candidates[i]

        if idx != prev_idx:
            dst = 0.0
            for j in range(dim):
                dst -= x[j] * X[idx, j]
            scores.push_back(pair[double, int](dst, idx))

        prev_idx = idx

    sort(scores.begin(), scores.end())

    return scores


cpdef query_all(double[::1] x, double[:, ::1] X, list trees, unsigned int n):
    """
    Return approximate nearest neighbours to point x from the model.
    """

    cdef unsigned int i
    cdef unsigned int dim, no_returns

    cdef vector[pair[double, int]] scores
    cdef vector[int] candidates
    cdef cnp.ndarray[int, ndim=1] result

    dim = X.shape[1]

    candidates = _get_candidates(x, trees, dim)
    scores = sort_candidates(x, X, dim, candidates)

    no_returns = min(n, scores.size())

    result = np.empty(no_returns, dtype=np.int32)

    for i in range(no_returns):
        result[i] = scores[i].second

    return result


cpdef get_candidates_all(double[::1] x, list trees, unsigned int dim, int number):
    """
    Return all candidate nearest neighbours to point x without a final brute force
    sorting step. The returned candidates are ordered by how many leaf nodes they
    share with the query point.
    """

    cdef unsigned int i, no_trees, no_candidates
    cdef int idx, prev_idx, idx_count
    cdef vector[pair[int, int]] scores
    cdef vector[int] candidates
    cdef int[::1] result

    candidates = _get_candidates(x, trees, dim)
    sort(candidates.begin(), candidates.end())

    prev_idx = -1
    idx_count = 0
    no_candidates = candidates.size()

    for i in range(no_candidates):

        idx = candidates[i]

        if idx == prev_idx:
            idx_count += 1
        else:
            if prev_idx != -1:
                scores.push_back(pair[int, int](-idx_count, prev_idx))
                idx_count = 1

        prev_idx = idx

    scores.push_back(pair[int, int](idx_count, prev_idx))
    sort(scores.begin(), scores.end())

    result_arr = np.empty(scores.size(), dtype=np.int32)
    result = result_arr

    for i in range(scores.size()):
        result[i] = scores[i].second

    return result_arr


cdef vector[int] _get_candidates(double[::1] x, list roots, int dim):
    """
    Get all memebers of x's leaf nodes.
    """

    cdef unsigned int i
    cdef unsigned int no_roots = len(roots)
    cdef Tree tree
    cdef Node *root
    cdef Node *leaf
    cdef vector[int] candidates

    for i in range(no_roots):
        tree = roots[i]
        root = tree.root
        leaf = query(root, tree.hyperplanes, x)
        candidates.insert(candidates.end(),
                          leaf.indices.begin(),
                          leaf.indices.end())

    return candidates


cdef class BArray:
    """
    Bytearray that keeps the track of the current offset.
    """

    cdef arr
    cdef unsigned int offset
    cdef char* char_arr

    def __init__(self, ba, unsigned int offset):

        self.arr = ba
        self.offset = offset
        self.char_arr = self.arr


cdef class Tree:
    """
    A random projection tree.

    It consists of two main data structures:
    - a root node which contains links to its child nodes
    - the set of hyperplanes used by the nodes
    """

    cdef Node *root
    cdef Hyperplanes* hyperplanes

    cdef unsigned int max_size
    cdef unsigned int dim

    def __init__(self, max_size, dim):

        self.max_size = max_size
        self.dim = dim
        self.hyperplanes = new_hyperplanes(dim)

        self.root = new_node(0)

    def make_tree(self, double[:, ::1] X):
        """
        Recursively build a random projection tree
        from X, starting at the root.
        """

        cdef unsigned int i

        for i in range(X.shape[0]):
            self.root.indices.push_back(i)

        make_tree(self.root, self.hyperplanes, X, self.max_size, 0)

    def index(self, idx, double[::1] x):
        """
        Add a point to the tree.
        """

        cdef Node *leaf

        leaf = query(self.root, self.hyperplanes, x)
        leaf.indices.push_back(idx)

    def shrink_to_size(self):

        slim_all(self.root)

    def get_leaf_nodes(self):
        """
        Yields pairs of (leaf_node_code, leaf point indices).
        """

        cdef unsigned int i
        cdef vector[Node*] leaves
        cdef list leaf_codes = []

        get_leaf_nodes(self.root, &leaves, leaf_codes, '')

        assert len(leaf_codes) == leaves.size()

        for i in range(len(leaf_codes)):
            yield leaf_codes[i], deref(leaves[i].indices)

    def serialize(self):
        """
        Serialize to a bytarray.
        """

        ba = bytearray()
        ba.extend(struct.pack('@II',
                              self.max_size,
                              self.dim))
        ba.extend(struct.pack('@II',
                              self.hyperplanes.dim,
                              self.hyperplanes.num))
        for i in range(self.hyperplanes.hyperplanes.size()):
            ba.extend(struct.pack(hyp_symbol,
                                  deref(self.hyperplanes.hyperplanes)[i]))

        write_node(self.root, ba)

        return ba

    def deserialize(self, byte_array, serialization_version):
        """
        Read tree from a bytearray.
        """

        cdef Node* node

        ba = BArray(byte_array, 0)

        if serialization_version != SERIALIZATION_VERSION:
            # Start reading according to the
            # previous protocol
            return self._deserialize_old(byte_array)

        self.max_size = struct.unpack_from('@I',
                                           ba.arr,
                                           offset=ba.offset)[0]
        ba.offset += uint_size
        self.dim = struct.unpack_from('@I',
                                      ba.arr,
                                      offset=ba.offset)[0]
        ba.offset += uint_size

        # Read hyperplanes
        dim = struct.unpack_from('@I',
                                 ba.arr,
                                 offset=ba.offset)[0]
        ba.offset += uint_size
        self.hyperplanes = new_hyperplanes(dim)
        num = struct.unpack_from('@I',
                                 ba.arr,
                                 offset=ba.offset)[0]
        ba.offset += uint_size
        self.hyperplanes.num = num

        for i in range(num * dim):
            self.hyperplanes.hyperplanes.push_back(
                struct.unpack_from(hyp_symbol,
                                   ba.arr,
                                   offset=ba.offset)[0])
            ba.offset += hyp_size

        # Read tree
        self.root = read_node(ba, self.dim)

    def _deserialize_old(self, byte_array):
        """
        Read tree from a bytearray serialized using the previous
        version of rpforest.
        """

        cdef Node* node

        ba = BArray(byte_array, 0)

        self.max_size = struct.unpack_from('@I',
                                           ba.arr,
                                           offset=ba.offset)[0]
        ba.offset += uint_size
        self.dim = struct.unpack_from('@I',
                                      ba.arr,
                                      offset=ba.offset)[0]
        ba.offset += uint_size

        # Read tree
        self.root = _read_node_old(ba, self.hyperplanes)

    def get_size(self):
        """
        Return the memory size of the tree, in bytes.
        """

        return (sizeof(Hyperplanes)
                + self.hyperplanes.hyperplanes.capacity()
                + get_size(self.root))

    def clear(self):
        """
        Remove all indexed points but retain the tree structure.
        """

        clear(self.root)

    def __dealloc__(self):

        del_node(self.root)
        free_hyperplanes(self.hyperplanes)


cdef void make_tree(Node *node, Hyperplanes* hyper, double[:, ::1] X,
                    unsigned int max_size, unsigned int depth):
    """
    Recursively build a random projection tree starting at node.
    """

    cdef int i
    cdef double dst
    cdef hyp* hyperplane
    cdef vector[double] dist

    cdef Node *left
    cdef Node *right

    if node.indices.size() <= max_size:
        slim_node(node)
        return

    # Allocate child nodes.
    left = new_node(depth + 1)
    right = new_node(depth + 1)

    # Create a new hyperplane if there is none
    # for the current depth
    if depth >= hyper.num:
        add_hyperplane(hyper)

    # Get the hyperplane
    hyperplane = get_hyperplane(hyper, depth)

    # Calculate the median cosine similarity
    # between the hyperplane and the points.
    for i in range(node.indices.size()):
        idx = deref(node.indices)[i]
        dst = dot(hyperplane,
                   X[idx, :],
                   hyper.dim)
        dist.push_back(dst)

    sort(dist.begin(), dist.end())
    node.median = dist[dist.size() / 2]

    # Split points at median similarity.
    for i in range(node.indices.size()):
        idx = deref(node.indices)[i]
        dst = dot(hyperplane,
                  X[idx, :],
                  hyper.dim)
        if dst <= node.median:
            left.indices.push_back(idx)
        else:
            right.indices.push_back(idx)

    # Bail out on a failed split.
    if left.indices.size() == 0 or right.indices.size() == 0:
        slim_node(node)
        del_node(left)
        del_node(right)
        return

    # Add point indices to children.
    add_descendants(node, left, right)
    slim_node(node)

    # Recursively split child subtrees.
    make_tree(left, hyper, X, max_size, depth + 1)
    make_tree(right, hyper, X, max_size, depth + 1)


cdef void get_leaf_nodes(Node *node, vector[Node*] *leaves, list leaf_codes, str code):
    """
    Retrieve all leaf nodes.
    """

    if node.n_descendants == 0:
        leaves.push_back(node)
        leaf_codes.append(code)
    else:
        get_leaf_nodes(node.left, leaves, leaf_codes, code + '0')
        get_leaf_nodes(node.right, leaves, leaf_codes, code + '1')


cdef void slim_all(Node *node):
    """
    Slim all nodes.
    """

    slim_node(node)

    if node.n_descendants != 0:
        slim_node(node.left)
        slim_node(node.right)


cdef packed struct Node:

    unsigned int depth
    unsigned char n_descendants
    hyp median

    vector[int] *indices

    Node *left
    Node *right


cdef struct Hyperplanes:
    unsigned int dim
    unsigned int num
    vector[hyp] *hyperplanes


cdef Hyperplanes* new_hyperplanes(unsigned int dim):

    cdef Hyperplanes* hyper

    hyper = <Hyperplanes *>malloc(sizeof(Hyperplanes))
    hyper.dim = dim
    hyper.num = 0
    hyper.hyperplanes = new vector[hyp]()

    return hyper


cdef void free_hyperplanes(Hyperplanes* hyper):

    del hyper.hyperplanes
    free(hyper)


cdef void add_hyperplane(Hyperplanes* hyper):

    cdef int i
    cdef hyp[::1] hyperplane

    hyper.num += 1

    # Generate the random hyperplane.
    hyperplane = np.random.randn(hyper.dim).astype(np.float32)
    hyperplane /= np.linalg.norm(hyperplane)

    hyper.hyperplanes.reserve(hyper.num * hyper.dim)

    for i in range(hyper.dim):
        hyper.hyperplanes.push_back(hyperplane[i])


cdef hyp* get_hyperplane(Hyperplanes* hyper, unsigned int row) nogil:

    return &(deref(hyper.hyperplanes)[hyper.dim * row])


cdef inline Node* new_node(unsigned int depth):
    """
    Allocate a new node.
    """

    cdef Node *node = <Node *>malloc(sizeof(Node))

    if node == NULL:
        raise MemoryError()

    node.n_descendants = 0
    node.indices = new vector[int]()
    node.depth = depth

    return node


cdef inline void slim_node(Node *node) noexcept nogil:
    """
    Deallocate the indices vector if the node is internal
    """

    cdef vector[int] swapped_indices

    if node.n_descendants > 0:
        if node.indices != NULL:
            del node.indices
            node.indices = NULL
    else:
        swapped_indices = vector[int](deref(node.indices))
        node.indices.swap(swapped_indices)


cdef inline void add_descendants(Node *node, Node *left, Node *right) noexcept nogil:

    node.n_descendants = 2
    node.left = left
    node.right = right


cdef void del_node(Node *node) noexcept nogil:
    """
    Free a node.
    """

    if node.n_descendants > 0:
        del_node(node.left)
        del_node(node.right)
    else:
        del node.indices

    free(node)


cdef void clear(Node *node) noexcept nogil:
    """
    Recursively remove all indexed points from node and
    its children.
    """

    cdef vector[int] swapped_indices

    if node.n_descendants == 0:
        node.indices.clear()
        swapped_indices = vector[int](deref(node.indices))
        node.indices.swap(swapped_indices)
    else:
        clear(node.left)
        clear(node.right)


cdef long get_size(Node *node) nogil:
    """
    Recursively get the size (in bytes) of node and
    its children.
    """

    cdef long size = 0

    size += sizeof(node)

    if node.n_descendants == 0:
        size += sizeof(vector[int]) + sizeof(int) * node.indices.size()
    else:
        size += sizeof(node)
        size += get_size(node.left)
        size += get_size(node.right)

    return size


cdef Node* query(Node *node, Hyperplanes* hyper, double[::1] x) nogil:
    """
    Recursively query the node and its children.
    """

    cdef double dst
    cdef hyp* hyperplane

    if node.n_descendants == 0:
        return node

    hyperplane = get_hyperplane(hyper, node.depth)

    dst = dot(hyperplane,
              x,
              hyper.dim)

    if dst <= node.median:
        return query(node.left, hyper, x)
    else:
        return query(node.right, hyper, x)


cdef void encode(Node *node, Hyperplanes* hyper, double[::1] x, list code, unsigned int dim):
    """
    Recursively encode x.
    """

    cdef double dst
    cdef hyp* hyperplane

    if node.n_descendants == 0:
        return

    hyperplane = get_hyperplane(hyper, node.depth)

    dst = dot(hyperplane,
              x,
              dim)

    if dst <= node.median:
        code.append('0')
        encode(node.left, hyper, x, code, dim)
    else:
        code.append('1')
        encode(node.right, hyper, x, code, dim)


# Serialisation and deserialisation
cdef void write_node(Node *node, ba):
    """
    Recursively write nodes to a bytearray.
    """

    cdef unsigned int i
    cdef hyp* hyperplane

    ba.extend(struct.pack('@B',
                          node.n_descendants))
    ba.extend(struct.pack('@I',
                          node.depth))
    ba.extend(struct.pack(hyp_symbol,
                          node.median))
    
    if node.n_descendants == 0:
        ba.extend(struct.pack('@I',
                              node.indices.size()))
        for i in range(node.indices.size()):
            ba.extend(struct.pack('@i',
                                  deref(node.indices)[i]))
    else:
        write_node(node.left, ba)
        write_node(node.right, ba)


cdef Node* read_node(BArray ba, unsigned int dim):
    """
    Recursively read nodes from bytearray.
    """

    # Using memcpy is orders of magnitude faster
    # than using struct.unpack_from.

    cdef unsigned int i, size
    cdef int idx
    cdef Node *node

    node = new_node(0)

    # Read number of descendants
    memcpy(&node.n_descendants, ba.char_arr + ba.offset, sizeof(unsigned char));
    ba.offset += uchar_size

    # Read number of descendants
    memcpy(&node.depth, ba.char_arr + ba.offset, sizeof(unsigned int));
    ba.offset += uint_size

    # Read median
    memcpy(&node.median, ba.char_arr + ba.offset, sizeof(hyp));
    ba.offset += hyp_size

    # Read leaf element ids if present
    if node.n_descendants == 0:
        memcpy(&size, ba.char_arr + ba.offset, sizeof(unsigned int));
        ba.offset += uint_size
        node.indices.reserve(size)
        for i in range(size):
            memcpy(&idx, ba.char_arr + ba.offset, sizeof(int));
            node.indices.push_back(idx)
            ba.offset += int_size
    else:
        # Read child nodes
        node.left = read_node(ba, dim)
        node.right = read_node(ba, dim)

    slim_node(node)

    return node


cdef Node* _read_node_old(BArray ba, Hyperplanes *hyper):
    """
    Recursively read nodes from bytearray created using
    serialization from the previous version.
    """

    # Using memcpy is orders of magnitude faster
    # than using struct.unpack_from.

    cdef unsigned int i, size
    cdef int idx
    cdef Node *node
    cdef hyp *hyperplane

    node = new_node(0)

    # Read number of descendants
    memcpy(&node.n_descendants, ba.char_arr + ba.offset, sizeof(unsigned char));
    ba.offset += uchar_size

    # Read median
    memcpy(&node.median, ba.char_arr + ba.offset, sizeof(hyp));
    ba.offset += hyp_size

    # Read leaf element ids if present
    if node.n_descendants == 0:
        memcpy(&size, ba.char_arr + ba.offset, sizeof(unsigned int));
        ba.offset += uint_size
        node.indices.reserve(size)
        for i in range(size):
            memcpy(&idx, ba.char_arr + ba.offset, sizeof(int));
            node.indices.push_back(idx)
            ba.offset += int_size

    else:
        # Allocate a new hyperplane per internal node
        add_hyperplane(hyper)

        # Set the node depth to point at
        # the new hyperplane
        node.depth = hyper.num - 1

        # Get the hyperplane
        hyperplane = get_hyperplane(hyper, node.depth)

        for i in range(hyper.dim):
            memcpy(&hyperplane[i], ba.char_arr + ba.offset, sizeof(hyp));
            ba.offset += hyp_size

        # Read child nodes
        node.left = _read_node_old(ba, hyper)
        node.right = _read_node_old(ba, hyper)

    slim_node(node)

    return node
