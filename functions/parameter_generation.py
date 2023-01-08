import numpy as np
from numpy.random import default_rng


def gen_random_dag(n, s, min_w_star=0.1):

    '''
    This script contains the functions used to generate the underlying coordination matrix, A, 
    in the sample evaluation section. This function produces an acyclic network which has edges
    placed with probability 0.5, and edge weights drawn from an exponential distribution with 
    parameter s. The minimum w_star represents the smallest probability of a single process not 
    coordinating, and is used to calculate the global normalisation constant.

    The normalisation constant is given by
    \[
        \left( \dfrac{\sum_j A_{i,j}}{(1-min_w_star)} \given \max{j} \sum_j A_{i,j}\right)$.
    \]

    '''

    # n = number of processes
    # returns adjacency matrix

    # gen random edge weights
    rng = default_rng()
    ew  = rng.exponential(s, (n,n))*np.random.randint(0,2,(n,n))

    # force dag
    dag = ew * np.triu(np.ones((n,n)), k=(np.floor(n/2)-1))
    col_sum = dag.sum(axis=0,keepdims=1) / (1-min_w_star); # divide by max column sum
    return dag / np.max(col_sum)
