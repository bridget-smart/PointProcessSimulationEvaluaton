"""
Preamble for most code and jupyter notebooks
@author: bridgetsmart
@notebook date: 5 Jul 2022
"""

import numpy as np, pandas as pd

import matplotlib.pyplot as plt, seaborn as sns
import matplotlib as mpl

import math, string, re, pickle, json, time, os, sys, datetime, itertools

from collections import Counter

import networkx as nx
import scipy
import bisect
from numpy.random import default_rng

def get_new_time(expected_time):
    # generate time between poisson events given the rate.
    # expected time = rate (expected time between events)

    return np.random.exponential(expected_time)

def sim_ct(rates, t_finish, t_current, current_state = 0):
    # simulating c(t) for a single process
    # c(t) is the underlying markov process which 
    # describes if a process is currently coordinating
    # or not.

    # rates = vector giving 
    #   [expected time to move into coordinated state,
    #    expected time to move into a non coordinated state]

    # t_finish = time to simulate events until
    # t_current = time to simulate events from

    # current_state = 0 # non coord is default

    transition_times = []
    t_dash = get_new_time(rates[current_state])
    t_current += t_dash
    current_state = np.mod(current_state+1,2)
    # generate exponential random variable with rate 1/to_coord
    while t_current<t_finish:
        transition_times.append(t_current)
        t_dash = get_new_time(rates[current_state])

        t_current += t_dash
        current_state = np.mod(current_state+1,2)


    return [k for k,v in Counter(transition_times).items() if v==1] # dict preserves order

def get_self_excited_times(rate, t_current, t_finish):
    # to generate times between t_current and
    # t_finish while a process is currently 
    # not coordinating (all events are self excited)

    t_current += get_new_time(rate)
    t_list = [t_current]
    while t_current<t_finish:
        t_list.append(t_current)
        t_current += get_new_time(rate)

    return t_list

def most_recent_time(list_times, current_time):
    hist = [x for x in list_times if x<current_time]
    if len(hist)>0:
        return hist[-1]
    else:
        return np.nan

def get_last_times(to_consider, times, current_time):
    # returns the most recent event times
    # from all processes in the list to_consider
    # the order of times matches the order in 
    # to consider

    return [most_recent_time(times[x], current_time) for x in to_consider]

def get_coord_delay_beta(C_rates, target, source, scale):
    # get the beta distributed delay between 
    # coordinated events
    # this returns a time delay for the target process, 
    # that is coordinating with the source 
    # process.

    # C_rates is a matrix describing the alpha and beta
    # for this delay. 

    return scale*scipy.stats.beta.rvs(*C_rates[source, target], size=1)[0]

def get_coordinated_time_beta(of_interest, G, A,prob_coord, t_current, time_importance_decay, decay_func, times, C_rates, scale):
    # given a target process (of_interest)
    # the coordination graph (G)
    # the coordination weightings (A)
    # the current time (t_current)
    # and a function which desscribes the effect of 
    # time difference (usually decaying),
    # this function gets the next event time for the
    # of_interest process.
    # note - this can occur BEFORE t_current

    # function to get next time | coordinating
    if prob_coord > default_rng().random(): # if we are coordinating

        # get list of coordinating nodes
        if A[of_interest,of_interest] >0:
            to_consider = list(nx.neighbors(G.reverse(),of_interest)) + [of_interest]# need to reverse to get dependencies # include self coordination
        else:
            to_consider = list(nx.neighbors(G.reverse(),of_interest))

        # print(to_consider)
        if len(to_consider)>0:
            # most_recent = times[to_consider], 0]
            last_event_times = get_last_times(to_consider, times, t_current)      
            is_nan = np.where(np.isnan(last_event_times))
            last_event_times = np.delete(last_event_times, is_nan)
            to_consider = np.delete(to_consider, is_nan)   

        if len(to_consider)>0:

            probs = []
            probs = [A[to_consider[i],of_interest] * time_importance_decay(t_current - last_event_times[i]) for i in range(len(to_consider))]# coordination factor * decay value * something to avoid underflow
            #normalise
            s=sum(probs)
            probs = [x/s for x in probs]

            g_zero = np.sum([x<0 for x in probs])
            if g_zero >0:
                print('Time decay function needs to be strictly greater than zero for all x.')
            # now draw from probs
            i=np.random.choice(len(probs), p=probs)
            return last_event_times[i] + get_coord_delay_beta(C_rates, of_interest, to_consider[i], scale), t_current+ get_coord_delay_beta(C_rates, of_interest, of_interest, scale), True # time to coordinate from + delay

        else:
            return t_current+ get_coord_delay_beta(C_rates, of_interest, of_interest, scale), t_current+ get_coord_delay_beta(C_rates, of_interest, of_interest, scale), False # new fake time

    else:
        return t_current+ get_coord_delay_beta(C_rates, of_interest, of_interest, scale), t_current+ get_coord_delay_beta(C_rates, of_interest, of_interest, scale), False

def get_coordinated_time(of_interest, G, A, prob_coord, t_current, time_importance_decay, decay_func, times, C_rates, scale):
    # given a target process (of_interest)
    # the coordination graph (G)
    # the coordination weightings (A)
    # the current time (t_current)
    # and a function which desscribes the effect of 
    # time difference (usually decaying),
    # this function gets the next event time for the
    # of_interest process.
    # note - this can occur BEFORE t_current

    # function to get next time | coordinating
    if prob_coord > default_rng().random(): # if we are coordinating

        # get list of coordinating nodes
        if A[of_interest,of_interest] >0:
            to_consider = list(nx.neighbors(G.reverse(),of_interest)) + [of_interest]# need to reverse to get dependencies # include self coordination
        else:
            to_consider = list(nx.neighbors(G.reverse(),of_interest))

        if len(to_consider)>0:
            # most_recent = times[to_consider], 0]
            last_event_times = get_last_times(to_consider, times, t_current)      
            is_nan = np.where(np.isnan(last_event_times))
            last_event_times = np.delete(last_event_times, is_nan)
            to_consider = np.delete(to_consider, is_nan)   

        if len(to_consider)>0:
            probs = []
            probs = [A[to_consider[i],of_interest] * time_importance_decay(t_current - last_event_times[i]) for i in range(len(to_consider))]# coordination factor * decay value * something to avoid underflow
            #normalise
            s=sum(probs)
            probs = [x/s for x in probs]
            g_zero = np.sum([x<0 for x in probs])

            # set nan to zero

            if g_zero >0:
                print('Time decay function needs to be strictly greater than zero for all x.')
            # now draw from probs
            i=np.random.choice(len(probs), p=probs)
            return last_event_times[i] + decay_func(), t_current+ decay_func(), True # time to coordinate from + delay

        else:
            return t_current+ decay_func(), t_current+ decay_func(), False # new fake time
    
    else:
        # not coordinating
        # no event
        # but still update current time
        return t_current+ decay_func(), t_current+ decay_func(), False


def times_coordinating(target, t_current, t_finish, prob_coord, G, A, time_importance_decay, decay_func, times, C_rates, scale, beta_decay):
    if beta_decay:
        t_list = []
        t_proposed, t_current, add = get_coordinated_time_beta(target, G, A,prob_coord,t_current, time_importance_decay, decay_func, times, C_rates, scale)

        if add:
            t_list.append(t_proposed)


        while t_current<t_finish:
            
            t_proposed, t_current, add = get_coordinated_time_beta(target, G, A, prob_coord, t_current, time_importance_decay, decay_func, times, C_rates, scale)

            if add:
                t_list.append(t_proposed)


        return t_list

    else:
        t_list = []
        t_proposed, t_current,add = get_coordinated_time(target, G, A, prob_coord, t_current, time_importance_decay, decay_func, times, C_rates, scale)


        if add:
            t_list.append(t_proposed)


        while t_current<t_finish:
            t_proposed, t_current, add = get_coordinated_time(target, G, A, prob_coord, t_current, time_importance_decay, decay_func, times, C_rates, scale)

            if add:
                t_list.append(t_proposed)
        return t_list

        

def get_current_state(target, c_t, current_time):
    # returns 0 if not coordinating
    # returns 1 otherwise
    ind_test = bisect.bisect_left(c_t[target], current_time)
    if ind_test == len(c_t[target]):
        return np.mod(ind_test,2), False
    else:
        return np.mod(ind_test,2), c_t[target][ind_test]

from collections.abc import Iterable

# need a function to flatten irregular list of lists
def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x