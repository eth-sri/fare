# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Jacob Schreiber <jmschreiber91@gmail.com>
#          Nelson Liu <nelson@nelsonliu.me>
#
# License: BSD 3 clause

# See _utils.pyx for details.

cimport numpy as cnp
from ..neighbors._quad_tree cimport Cell

ctypedef cnp.npy_float32 DTYPE_t          # Type of X
ctypedef cnp.npy_float64 DOUBLE_t         # Type of y, sample_weight
ctypedef cnp.npy_intp SIZE_t              # Type for indices and counters
ctypedef cnp.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef cnp.npy_uint32 UINT32_t          # Unsigned 32 bit integer
ctypedef cnp.npy_uint64 UINT64_t          # Unsigned 64 bit integer
ctypedef cnp.npy_bool BOOL_t              # Bool
from libc.stdio cimport printf
from libc.math cimport round

# Categorical 

cdef inline void print_partition(CategoricalPartition partition, int maxval) nogil:
    printf("Partition for feature with values in [1, %d]:\n", maxval)
    cdef SIZE_t i = 1
    while i <= maxval:
        pres = bitset_get(partition.present, <INT32_t>i)
        side = bitset_get(partition.side, <INT32_t>i)
        if not pres:
            printf("_")
        else:
            if side == 0:
                printf("L")
            else:
                printf("R")
        if i % 64 == 63:
            printf("|\n")
        i += 1
    printf("\n")

cdef inline BOOL_t goes_left(DOUBLE_t val, BOOL_t is_cat, DOUBLE_t threshold, CategoricalPartition* partition) nogil:
    # ignore presence!
    if is_cat: 
        #assert bitset_get(partition.present, <INT32_t>val) == 1
        if bitset_get(partition.side, <INT32_t>round(val)) == 0:
            return True
        else:
            return False 
    else:
        if val <= threshold:
            return True 
        else:
            return False

cdef inline int bitset_get(UINT64_t* bs, int idx) nogil:
    b_idx = idx // 64
    idx = idx % 64 
    return (bs[b_idx] >> idx) & (<UINT64_t> 1)

cdef inline void bitset_set1(UINT64_t* bs, int idx) nogil:
    b_idx = idx // 64
    idx = idx % 64 
    bs[b_idx] = bs[b_idx] | ((<UINT64_t> 1) << idx)
    
cdef inline void bitset_set0(UINT64_t* bs, int idx) nogil:
    b_idx = idx // 64
    idx = idx % 64 
    bs[b_idx] = bs[b_idx] & ~((<UINT64_t> 1) << idx)

cdef inline void bitset_all0(UINT64_t* bs) nogil:
    for i in range(8):
        bs[i] = 0
        
cdef inline void bitset_all1(UINT64_t* bs) nogil:
    for i in range(8):
        bs[i] = ~(<UINT64_t> 0)

cdef struct CategoricalPartition:
    UINT64_t side[8]
    UINT64_t present[8]

cdef struct Node:
    # Base storage structure for the nodes in a Tree object

    SIZE_t left_child                    # id of the left child of the node
    SIZE_t right_child                   # id of the right child of the node
    SIZE_t feature                       # Feature used for splitting the node

    # Only one of the following two will be used (based on `feature`)
    DOUBLE_t threshold           # Threshold at the node (continuous)
    CategoricalPartition partition       # Partition of categoricals

    DOUBLE_t impurity                    # Impurity of the node (i.e., the value of the criterion)
    SIZE_t n_node_samples                # Number of samples at the node
    DOUBLE_t weighted_n_node_samples     # Weighted number of samples at the node
    
   
cdef struct SplitRecord:
    # Data to track sample split
    SIZE_t feature         # Which feature to split on.
    SIZE_t pos             # Split samples array at the given position,
                           # i.e. count of samples below threshold for feature.
                           # pos is >= end if the node is a leaf.

    double threshold                # Threshold to split at (cont)
    CategoricalPartition partition          # Partition of categoricals

    double improvement     # Impurity improvement given parent node.
    double impurity_left   # Impurity of the left split.
    double impurity_right  # Impurity of the right split.


cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF


# safe_realloc(&p, n) resizes the allocation of p to n * sizeof(*p) bytes or
# raises a MemoryError. It never calls free, since that's __dealloc__'s job.
#   cdef DTYPE_t *p = NULL
#   safe_realloc(&p, n)
# is equivalent to p = malloc(n * sizeof(*p)) with error checking.
ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (DTYPE_t*)
    (SIZE_t*)
    (unsigned char*)
    (WeightedPQueueRecord*)
    (DOUBLE_t*)
    (DOUBLE_t**)
    (Node*)
    (Cell*)
    (Node**)
    (UINT64_t*)

cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) nogil except *


cdef cnp.ndarray sizet_ptr_to_ndarray(SIZE_t* data, SIZE_t size)


cdef SIZE_t rand_int(SIZE_t low, SIZE_t high,
                     UINT32_t* random_state) nogil


cdef double rand_uniform(double low, double high,
                         UINT32_t* random_state) nogil


cdef double log(double x) nogil

# =============================================================================
# WeightedPQueue data structure
# =============================================================================

# A record stored in the WeightedPQueue
cdef struct WeightedPQueueRecord:
    DOUBLE_t data
    DOUBLE_t weight

cdef class WeightedPQueue:
    cdef SIZE_t capacity
    cdef SIZE_t array_ptr
    cdef WeightedPQueueRecord* array_

    cdef bint is_empty(self) nogil
    cdef int reset(self) nogil except -1
    cdef SIZE_t size(self) nogil
    cdef int push(self, DOUBLE_t data, DOUBLE_t weight) nogil except -1
    cdef int remove(self, DOUBLE_t data, DOUBLE_t weight) nogil
    cdef int pop(self, DOUBLE_t* data, DOUBLE_t* weight) nogil
    cdef int peek(self, DOUBLE_t* data, DOUBLE_t* weight) nogil
    cdef DOUBLE_t get_weight_from_index(self, SIZE_t index) nogil
    cdef DOUBLE_t get_value_from_index(self, SIZE_t index) nogil


# =============================================================================
# WeightedMedianCalculator data structure
# =============================================================================

cdef class WeightedMedianCalculator:
    cdef SIZE_t initial_capacity
    cdef WeightedPQueue samples
    cdef DOUBLE_t total_weight
    cdef SIZE_t k
    cdef DOUBLE_t sum_w_0_k            # represents sum(weights[0:k])
                                       # = w[0] + w[1] + ... + w[k-1]

    cdef SIZE_t size(self) nogil
    cdef int push(self, DOUBLE_t data, DOUBLE_t weight) nogil except -1
    cdef int reset(self) nogil except -1
    cdef int update_median_parameters_post_push(
        self, DOUBLE_t data, DOUBLE_t weight,
        DOUBLE_t original_median) nogil
    cdef int remove(self, DOUBLE_t data, DOUBLE_t weight) nogil
    cdef int pop(self, DOUBLE_t* data, DOUBLE_t* weight) nogil
    cdef int update_median_parameters_post_remove(
        self, DOUBLE_t data, DOUBLE_t weight,
        DOUBLE_t original_median) nogil
    cdef DOUBLE_t get_median(self) nogil
