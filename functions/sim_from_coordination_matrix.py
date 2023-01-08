import numpy as np, pandas as pd

import math, string, re, pickle, json, time, os, sys, datetime, itertools


import networkx as nx
from functions.single_process_simulation import *


def from_coordination_matrix(A, c_t_params, lambdas, coord_decay_params, scale, time_importance_decay, decay_func, probs_coord, t_start, t_burnin, t_end, t_del, beta_decay):

    '''
    Function to generate a set of event times given the input parameters.

    A is a $n \times n$ weighted adjacency matrix which describes the underlying coordination structure for directly coordinating events. Since we use a topological ordering to populate the event times for the $n$ processes, we require that this adjacency matrix defines a directed acyclic graph. This matrix forms the basis for all coordination within the network, and represents the ground truth for directly coordinating relationships. A matrix with more edges represents a system where many processes directly coordinate, whereas the presence of paths represents processes which coordinate in sequence. 

    probs_coord is a $1 \times n$ vector that describes the likelihood of each process producing a coordinated event when in a coordinating state. This is set so $1 = \sum_j A_{i,j} + w^\star_i$ for all $i$, or that the column sum, plus the respective $w^{\star}_i$ is equal to one. For parameter generation, the minimum value of $w^{\star}_i$ is set, and the values in $A$ are normalised by the value $ \left( \dfrac{\sum_j A_{i,j}}{(1-w^{\star})} \given \max{j} \sum_j A_{i,j}\right)$.

    c_t_params is a vector of tuples, with row $i$ containing $(\dfrac{1}{a_{0,1}^{(i)}}, \dfrac{1}{a_{1,0}^{(i)}})$. $(a_{0,1}^{(i)}, a_{1,0}^{(i)})$ are the parameters which define the rates of the continuous time hidden Markov process which is used to generate the coordinating state of process $i$. In the hidden Markov process, transitions from a non-coordinating state to a coordinating state occur with rate $a_{0,1}$ and transitions back occur at rate $a_{1,0}$. The sequence of times generated from this hidden Markov process is denoted $c(t)^{(i)}$ and describes the state at time $t$ for process $i$. If a process is always in a coordinating state, this can be represented by setting $a_{0,1}>0$ and $a_{1,0}=0$. If a process moves between a coordinating state and a non-coordinating state quickly, both $a_{0,1}$ and $a_{1,0}$ should be large. If a process goes through long periods of coordinating and long periods of not coordinating, both rates should be small. The expected time which the process remains in each state is given by $\dfrac{1}{a_{n,m}}$, so these values can be experimentally obtained if data is available. All processes start in a non-coordinating state during the burn-in period, so if it is necessary for each process to always coordinate after this point, an additional burn-in period may be needed.
    
    lambdas is the vector of rates which are used to generate the self-excited event times. These times are generated from a Poisson process with rate $\lambda^{(i)}$. The expected time between events is given by $\dfrac{1}{\lambda^{(i)}}$. If this process produces self-excited events at a rate much higher than is possible for coordinated events, for example if the time delay for a coordinated event is very long, periods of coordination will be characterised by fewer events. It can be important to make sure these two types of events occur at similar overall rates, if periods of coordination are not easily distinguished by the event rate in real data.

    decay_func is the coordinated delay function used to simulate the time delay for a coordinated event in target process $i$ using source process $j$. If not specified by the user, $B$ and $s$ are used instead.
    
    coord_decay_params is a $n \times n$ matrix of tuples used to calculate the beta distributed delay between a parent and child coordinated event pair. $B_{(i,j)} = (\alpha^{(i,j)},\beta^{(i,j)})$ which are the parameters for an event in process $i$ which has a parent event in process $j$. The beta distribution was chosen as the two parameters afford a large amount of control over the shape of the distribution. This parameter is only used if a custom coordinated event decay is not specified. 
    
    scale is the scaling constant for beta distributed coordination time decay, to ensure the delays are on a meaningful scale. This can be used to ensure event occurrence rates in both coordinated and non-coordinated periods are similar or influence the size of a coordinated decay without altering the shape of the beta-distribution.
    
    time_importance_decay For a coordinated event, the parent event is chosen using a normalised score for each possible parent process. The score for process $j$ is calculated using the product of the time importance decay function applied to the time from the most recent event in process $j$ and the weight of the directed coordination edge between process $i$ and process $j$. This function impacts how the memory of a process impacts coordinating events. If $d(t)$ is chosen as a exponential function, more recent events are given a higher weighting, but all possible parent events are considered. If $d(t)$ is a threshold function, coordination can only occur within a specified window. If $d(t)$ is chosen to be uniform, the elapsed time does not impact the parent process.
    
    t_start The start time. No events can occur before this time.
    
    t_end The end time after which no events can occur.
    
    t_burnin The burn-in time is used to ensure all processes have generated some events which can be used as parent events. This time needs to be larger than $T_{\text{start}}$ and smaller than $T_{\text{end}}$. Between $T_{\text{start}}$ and $T_{\text{burn-in}}$, only self-excited events are permitted. 

    t_del Small time delta used to jump forward if stuck.

    beta_decay Boolean value which dictates if the beta coordination delay or a custom specified function (decay_func) is used.
    '''
    # A - coordination graph with weights

    # set up graph
    # remove diag and remove very last row
    G=nx.from_numpy_matrix(A, create_using=nx.DiGraph)
    n = A.shape[1]

    # fill times with burn in samples
    times = {i: get_self_excited_times(lambdas[i], t_start, t_burnin) for i in range(n)}

    t_current = t_burnin

    # first get c_t times
    c_t = {i: sim_ct(c_t_params[i], t_end, t_current, current_state = 0) for i in range(n)}

    for target in list(nx.topological_sort(G)): # produces generator which gives order to process nodes
        times_to_add = []
        t_current = t_burnin
        prob_coord = probs_coord[target]
        while t_current < t_end:
            times_coord = True
            # need to generate more times

            # check if coordinating or not
            state, time_until = get_current_state(target, c_t, t_current)

            if time_until:
                if state == 0: # not coordinating
                    # print('not coord')
                    times_to_add.append(get_self_excited_times(lambdas[target], t_current, time_until))

                elif state == 1: # coordinating
                    if len(list(nx.neighbors(G.reverse(),target)))>0:
                        # print('coord')

                        # print(f'beta decay is {beta_decay}')
                        times_coord = times_coordinating(target, t_current, time_until, prob_coord, G, A, time_importance_decay, decay_func, times, coord_decay_params, scale, beta_decay)
                        times_to_add.append(times_coord)
                    
                # flatten times_to_add
                if times_coord:
                    # print('joining up')
                    times_to_add = list(flatten(times_to_add))
            
                t_current = time_until+t_del
                
            # need to be careful to ensure don't end up in weird loop without ending becuase time until can't be more than t_end
        
            else: # we are in the last portion
                time_until = t_end
                if state == 0: # not coordinating
                    times_to_add.append(get_self_excited_times(lambdas[target], t_current, time_until))

                elif state == 1: # coordinating
                    times_coord = times_coordinating(target, t_current, time_until, prob_coord, G, A, time_importance_decay, decay_func, times, coord_decay_params, scale, beta_decay)
                    times_to_add.append(times_coord)
                            # flatten times_to_add

                if times_coord:
                    times_to_add = list(flatten(times_to_add))
                
                t_current = t_end
        t_list = times[target]
        t_list.append(times_to_add)
        t_list = list(flatten(t_list))
        times[target] = t_list
    # for any process, will always start in self excited

    return times, list(c_t.values())

