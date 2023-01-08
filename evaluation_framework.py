
import numpy as np, pandas as pd

import matplotlib.pyplot as plt, seaborn as sns
import matplotlib as mpl

from numpy.random import default_rng

import math, string, re, pickle, json, time, os, sys, datetime, itertools


from functions.time_coord_func import *
from functions.sim_from_coordination_matrix import *
from functions.evaluating_methods import *
from functions.parameter_generation import *

# set up default values
defaults = {}

n=10
defaults['n'] = n

# distribution of edge weights
dist_edge_weight_range = np.exp(-np.arange(0,5))
m_iter = 5 # number of times we generate the random graph model

# VARIABLES TO SET UP
# specification of coordination matrix - iterate through
min_w_star = 0.1
defaults['A'] = np.array([[0, 0, 0, 0, 0.26096396,0.02550826, 0.19027154, 0.09220806, 0, 0],[0, 0, 0, 0, 0, 0, 0.1724357 , 0.04033744, 0.03912118, 0],[0, 0, 0, 0, 0,0, 0.49473696, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0.7674545 , 0.34607817, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0.21403631],[0, 0, 0, 0, 0,0, 0, 0, 0, 0.05186133],[0, 0, 0, 0, 0,0, 0, 0, 0, 0],[0, 0, 0, 0, 0,0, 0, 0, 0, 0],[0, 0, 0, 0, 0,0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

defaults['probs_coord'] =1-defaults['A'].sum(axis=0)

# parameters for c(t)
c_param = [5,5]
c_t_params = np.empty(n, dtype=object)
c_t_params.fill(c_param)

defaults['c_t_params'] = c_t_params



# self exciting rates
lambdas = 3*np.ones((n))
defaults['lambdas'] = lambdas

# set up coordination decays
beta_param = (1,3)
coord_decay_params = np.empty((n,n), dtype=object)
coord_decay_params.fill(beta_param)
defaults['beta'] = beta_param
defaults['coord_decay_params'] = coord_decay_params
defaults['scale'] = "" # placeholder as not using beta decay func

# time importance decay function
time_importance_decay = {'exp' : lambda x: 100*np.exp(-1/2*x)} # make sure large enough to avoid underflow

defaults['time_importance_decay'] = time_importance_decay
defaults['decay_func'] = {'exp' : lambda : 5}

# settines
defaults['t_burnin'] = 100
defaults['t_start'] = 0
defaults['t_end'] = 2000 # 100 time units we are about

defaults['t_del'] = 1e-6

###################################################################################
# iteration values 
# None if don't want to vary
# if want to vary, need to provide a list
iter_values = {}

# iter_values['A'] = None
# iter_values['c_param'] = None
# iter_values['lambdas'] = [lambdas]

# c_p_list = []
# b_list = []
# beta_param = (1,1)
# b_list.append(beta_param)
# coord_decay_params = np.empty((n,n), dtype=object)
# coord_decay_params.fill(beta_param)
# c_p_list.append(coord_decay_params)

# beta_param = (1,2)
# b_list.append(beta_param)
# coord_decay_params = np.empty((n,n), dtype=object)
# coord_decay_params.fill(beta_param)
# c_p_list.append(coord_decay_params)

# beta_param = (1,3)
# b_list.append(beta_param)
# coord_decay_params = np.empty((n,n), dtype=object)
# coord_decay_params.fill(beta_param)
# c_p_list.append(coord_decay_params)

# beta_param = (1,4)
# b_list.append(beta_param)
# coord_decay_params = np.empty((n,n), dtype=object)
# coord_decay_params.fill(beta_param)
# c_p_list.append(coord_decay_params)
# iter_values['coord_decay_params'] = c_p_list

# HMM Parameters

iter_values['A'] = [gen_random_dag(n,s, min_w_star) for s in dist_edge_weight_range for _ in range(m_iter)]
iter_values['lambdas'] = [x*np.random.random((n))+0.2 for x in np.arange(0,21,2)]
# iter_values['beta'] = b_list

# iter_values['scale'] = None
# all functions must be strictly greater than zero
# iter_values['time_importance_decay'] = {'unif': lambda x: 20*x*(x>0)+0.01,
                                        # 'exp': lambda x: 1000*np.exp(-2*x)}

def return_one():
    return 1
    
iter_values['decay_func'] = {'unif (0,10)': lambda : 10*default_rng().random(),
                                        'exp mean 2': lambda : default_rng().exponential(1/2),
                                        'exp mean 5': lambda : default_rng().exponential(1/5),
                                        'Fixed (1 unit)' : lambda : 1,
                                        'Fixed (5 units)' : lambda : 5,
                                        'Fixed (10 units)' : lambda : 10}


possible_c_param = [[i,j] for i in range(1,21,2) for j in range(1,21,2)]
iter_values['c_t_params'] = [[x for _ in range(n)] for x in possible_c_param]

# iter_values['t_burnin'] = None
# iter_values['t_start'] = None
# iter_values['t_end'] = None
# iter_values['t_del'] = None

# List of functions which are to be evaluated
metric_func = [ median_time_delta, var_time_delta, cooccurance_count_1, cooccurance_count_5, cooccurance_count_10, exp_time_delta, time_agnostic] #exp_time_delta, [transfer_ent_est]#
names = ['median_time_delta', 'var_time_delta', 'cooccurance_count_1', 'cooccurance_count_5', 'cooccurance_count_10', 'exp_time_delta', 'time_agnostic_entropy'] #'exp_time_delta', 'transfer_entropy'
n_iter = 200
comparison_type = 'both'
ind = 0



res = assess_metric(metric_func, iter_values, defaults, beta_decay=False, comparison_type = 'both', n_iter=n_iter)



#### Save results
m_f = []
val = []
dc_sim = []
gf_sim = []
tags = []
t_lengths = []
for y in res:
    tag = y[1]
    for t in y[0]:   
        val.append([np.var(t[1]['A'][np.nonzero(t[1]['A'])]),t[1]['c_t_params'][0][0], t[1]['c_t_params'][0][1], t[1]['decay_func_name'], np.mean(t[1]['lambdas'])])
        t_lengths.append(np.mean(t[2]))
        dc_sim.append(t[3])
        gf_sim.append(t[4])
        m_f.append(t[0])
        tags.append(tag)


# df_plot
# 'metric_func'
df_plot = pd.DataFrame()#np.array(val), columns = col_names)
df_plot['metric_func'] = m_f
df_plot['N'] = t_lengths
df_plot['to_target'] = tags
df_plot['Direct Coordination Similarity'] = dc_sim
df_plot['Direct Coordination Similarity'] = df_plot['Direct Coordination Similarity'].astype('float')
df_plot['Generating Function Similarity'] = gf_sim
df_plot['Generating Function Similarity'] = df_plot['Generating Function Similarity'].astype('float')


# get out var
df_plot['Edge weight variance'] = np.array(val).T[0,:]
df_plot['Edge weight variance'] = df_plot['Edge weight variance'].astype('float')
df_plot['Average time in non-coordinating state'] = np.array(val).T[1,:]
df_plot['Average time in non-coordinating state'] = df_plot['Average time in non-coordinating state'].astype('int')
df_plot['Average time in coordinating state'] = np.array(val).T[2,:]
df_plot['Average time in coordinating state'] = df_plot['Average time in coordinating state'].astype('int')
df_plot['Time delay function'] = np.array(val).T[3,:]
df_plot['Time delay function'] = df_plot['Time delay function'].astype('string')
df_plot['Mean Rate of self-excited events'] = np.array(val).T[4,:]
df_plot['Mean Rate of self-excited events'] = df_plot['Mean Rate of self-excited events'].astype('float')


# save
with open(f'all_results.pkl', 'wb') as f:
    pickle.dump(df_plot, f)