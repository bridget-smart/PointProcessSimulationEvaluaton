from scipy.spatial import distance_matrix
from functions.parameter_generation import *
from functions.sim_from_coordination_matrix import *
from time import sleep
import os
from itertools import combinations


def matrix_similarity(A,B, measure = 'lr', norm = True):
    '''
    Returns similarity measure between two matrices A and B.
    '''

    # normalise
    if norm:
        A_ = A/(np.sum(A)) # normalise by sum or maximum element
        B_ = B/(np.sum(B))
    else:
        A_ = A
        B_ = B

    try:
        if measure == 'lr':
            return (np.corrcoef(A_.flatten(), B_.flatten())[0,1])**2

        elif measure == 'frobenius-norm':
            # Frobenius norm
            return np.linalg.norm(A_ - B_,ord='fro') 

        elif measure == 'max_norm':
            return np.max(A_) - np.max(B_)

        else:
            raise ValueError

    except ValueError:
        print("Invalid Input - measure needs to be lr, 2,1-norm or max_norm.")


# Generating function similarity functions

def rate_coord_event_times(event_times, coord_periods, start_time, end_time):
    '''
    Function to calculate the pair of rates for each process when it is in a coordinating
    or non-coordinating state. This is used as the ground truth for similarity between generating processes
    in the evaluation framework.
    '''
    coord_periods.append(end_time)
    rate_coord = 0
    rate_not_coord = 0

    interval_start = start_time
    coord_rate_elapsed = 0
    coord_rate_event_count = 0
    not_coord_rate_elapsed = 0
    not_coord_rate_event_count = 0

    current_state = 0

    for c_time in coord_periods:
        if current_state: # coordinating
            coord_rate_event_count += len([x for x in event_times if  interval_start<= x <c_time])
            coord_rate_elapsed += c_time - interval_start
            current_state = 0

        else: # not coord
            not_coord_rate_event_count += len([x for x in event_times if interval_start<=x <c_time])
            not_coord_rate_elapsed += c_time - interval_start
            current_state = 1

        interval_start = c_time

    if coord_rate_elapsed !=0:
        rate_est_coord = coord_rate_event_count / coord_rate_elapsed
    else:
        rate_est_coord = 0
    
    if not_coord_rate_elapsed !=0:
        rate_est_not_coord = not_coord_rate_event_count / not_coord_rate_elapsed
    else:
        rate_est_not_coord =0

    return rate_est_coord, rate_est_not_coord
    # return coord_rate_event_count / (end_time-start_time) +  not_coord_rate_event_count / (end_time-start_time) # time weighted rate average


def distances(list_of_lists):
    return distance_matrix(list_of_lists,list_of_lists)


def generating_similarity_matrix(times, coords, start_time, end_time):

    # Provides alternative to A matrix 

    processes = times.keys()
    rates = []
    for process in processes:
        rates.append(rate_coord_event_times(times[process], coords[process], start_time, end_time))

    return  1./(distances(rates)+0.01)



# evaluation

def one_set_run(iter_values, defaults, to_target, metric_func_list, beta_decay = True, comparison_type = 'direct coordination',n_iter = 10):
    # run for one set of modified variables
    # to_target = variable to modify
    # function to run an iteration cycling through values



    res = []

    # for target variable
    # remove from list of all
    # get all variables in a line
    # iterate through target variables
    l = len(iter_values[to_target])-1 # number of options to cycle through

    if beta_decay:
        while l>=0:
            to_use = {}

            for var in ['A', 'c_t_params', 'lambdas', 'coord_decay_params', 'scale', 'time_importance_decay','probs_coord','t_burnin', 't_start', 't_end', 't_del','n']:
                if var == to_target:
                    if var == 'time_importance_decay':
                        # this ones a dict
                        to_use[var] = list(iter_values[var].values())[l]
                        to_use['time_importance_decay_name'] =  list(iter_values[var].keys())[l]

                    elif var == 'decay_func':
                        # this ones a dict
                        to_use[var] = list(iter_values[var].values())[l]
                        to_use['decay_func_name'] =  list(iter_values[var].keys())[l]

                    else:
                        to_use[var] = iter_values[var][l]

                elif var == 'decay_func':
                    # this ones a dict
                    to_use[var] = list(defaults[var].values())[0]
                    to_use['decay_func_name'] =  list(defaults[var].keys())[0]

                else:
                    if var == 'time_importance_decay':
                        # this ones a dict
                        to_use[var] = list(defaults[var].values())[0]
                        to_use['time_importance_decay_name'] =  list(defaults[var].keys())[0]
                    else:
                        to_use[var] = defaults[var]

            to_use['decay_func'] = None

            l = l - 1

            res = run_compare_one_run(to_use, beta_decay, metric_func_list, comparison_type, res, n_iter)

    else:
        while l>=0:
            to_use = {}

            for var in ['A', 'c_t_params', 'lambdas', 'time_importance_decay','decay_func','probs_coord','t_burnin', 't_start', 't_end', 't_del','n']:
                if var == to_target:
                    if var == 'time_importance_decay':
                        # this ones a dict
                        to_use[var] = list(iter_values[var].values())[l]
                        to_use['time_importance_decay_name'] =  list(iter_values[var].keys())[l]

                    elif var == 'decay_func':
                        # this ones a dict
                        to_use[var] = list(iter_values[var].values())[l]
                        to_use['decay_func_name'] =  list(iter_values[var].keys())[l]

                    else:
                        to_use[var] = iter_values[var][l]

                elif var == 'decay_func':
                    # this ones a dict
                    to_use[var] = list(defaults[var].values())[0]
                    to_use['decay_func_name'] =  list(defaults[var].keys())[0]

                else:
                    if var == 'time_importance_decay':
                        # this ones a dict
                        to_use[var] = list(defaults[var].values())[0]
                        to_use['time_importance_decay_name'] =  list(defaults[var].keys())[0]
                    else:
                        to_use[var] = defaults[var]

            to_use['coord_decay_params']=None
            to_use['scale'] = None

            l = l - 1

            res = run_compare_one_run(to_use, beta_decay, metric_func_list, comparison_type,  res, n_iter)

    
    return res

def filter_list(li, thres_l, thres_u):
    return [x for x in li if thres_u>=x>=thres_l]




def run_compare_one_run(to_use, beta_decay, metric_func_list, comparison_type, res, n_iter):
    # metric_func - function which returns a pairwise coordination similarity measure
    # time importance decay function
    n = to_use['A'].shape[0]
    for _ in range(n_iter):
        times, coords = from_coordination_matrix(to_use['A'], 
                                        to_use['c_t_params'], 
                                        to_use['lambdas'], 
                                        to_use['coord_decay_params'],
                                        to_use['scale'], 
                                        to_use['time_importance_decay'], 
                                        to_use['decay_func'],
                                        to_use['probs_coord'],
                                        to_use['t_start'], 
                                        to_use['t_burnin'], 
                                        to_use['t_end'], 
                                        to_use['t_del'],
                                        beta_decay)

            # use times matrix to recreate A (may be scaled due to choice of metric_func)
        times = {k:filter_list(v, to_use['t_burnin'], to_use['t_end']) for k,v in times.items()}
        t_lengths = [len(x) for k,x in times.items()]
        for metric_func in metric_func_list:
            B = np.zeros((n,n))
            for comb in list(combinations(times.keys(), 2)):
                i=comb[0]
                j=comb[1]
                source = np.sort(times[i])
                target = np.sort(times[j])
                B[i,j] = metric_func(source, target)

            # comparison matrix
            if comparison_type == 'direct coordination':
                m_comp = to_use['A']
                res.append([metric_func.__name__,to_use,t_lengths, matrix_similarity(m_comp,B, measure = 'lr', norm = True)])
            elif comparison_type == 'generating similarity':
                m_comp = generating_similarity_matrix(times, coords, to_use['t_burnin'], to_use['t_end'])
                res.append([metric_func.__name__,to_use, t_lengths, matrix_similarity(m_comp,B, measure = 'lr', norm = True)])
            elif comparison_type == 'both':
                m1_comp = to_use['A']
                m2_comp = generating_similarity_matrix(times, coords, to_use['t_burnin'], to_use['t_end'])
                res.append([metric_func.__name__,to_use, t_lengths, matrix_similarity(m1_comp,B, measure = 'lr', norm = True), matrix_similarity(m2_comp,B, measure = 'lr', norm = True)])
            else:
                print('Comparison type is not valid, please use either \'direct coordination\' or \'generating similarity\'.')
    return res

def assess_metric(metric_func_list, iter_values, defaults, beta_decay = True, comparison_type = 'direct coordination', n_iter = 10):
    # each parameter of name_number
    # should be a np.arange, giving the range of values you 
    # want to check

    # set 'default values'

    res = []
    for to_target in iter_values.keys():
        res.append([one_set_run(iter_values, defaults, to_target, metric_func_list, beta_decay, comparison_type = comparison_type, n_iter = n_iter),to_target])

    return res



def check_function_existence(path, fn):
    '''
    Function to check if file exists at a given path.
    '''
    return fn in os.listdir(path)
    
# General Matrix similarities
def wait_on_file(path, fn, seconds):
    while not(check_function_existence(path, fn)):
    # wait
        sleep(seconds)