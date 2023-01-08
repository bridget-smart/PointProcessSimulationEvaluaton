import numpy as np
from ProcessEntropy.SelfEntropy import self_entropy_rate


def get_match(list_times, target_time):
    # Gets the most recent time in the list of times which preceeds the target time.

    if len(list_times[list_times <target_time])>0:
        t=len(list_times[list_times<target_time])-1
        return(t, list_times[t])
    else:
        return(np.nan,np.nan)
    
def median_time_delta(list_times,target_times):
    
    time_deltas = []
    
#     for t in p2:
#         if ( not np.isnan(t)) :
#             time_deltas.append(t - get_match(p1,t)[1])
#         else:
#             pass

    for t in target_times:
        time_deltas.append(t - get_match(np.array(list_times),t)[1])

    return(1/np.nanmedian(time_deltas)) # 1/ so larger is stronger

def var_time_delta(list_times,target_times):
    
    time_deltas = []
    
#     for t in p2:
#         if ( not np.isnan(t)) :
#             time_deltas.append(t - get_match(p1,t)[1])
#         else:
#             pass

    for t in target_times:
        time_deltas.append(t - get_match(np.array(list_times),t)[1])

    return(1/np.nanvar(time_deltas)) # 1/ so larger is stronger

def exp_time_delta(list_times,target_times):
    # both list and target need to be np arrays
    time_deltas = []
    
#     for t in p2:
#         if ( not np.isnan(t)) :
#             time_deltas.append(t - get_match(p1,t)[1])
#         else:
#             pass

    for t in target_times:
        time_deltas.append(np.exp(-(t - get_match(np.array(list_times),t)[1])))

    return(1/np.nanmean(time_deltas))# 1/ so larger is stronger


def cooccurance_count_1(list_times,target_times):
    # both list and target need to be np arrays
    count = 0 

    for t in target_times:
        interevent_time = t - get_match(np.array(list_times),t)[1]
        if interevent_time <= 1:
            count +=1

    return(count/ len(target_times))


def cooccurance_count_5(list_times,target_times):
    # both list and target need to be np arrays
    count = 0 

    for t in target_times:
        interevent_time = t - get_match(np.array(list_times),t)[1]
        if interevent_time <= 5:
            count +=1

    return(count / len(target_times))


def cooccurance_count_10(list_times,target_times):
    # both list and target need to be np arrays
    count = 0 

    for t in target_times:
        interevent_time = t - get_match(np.array(list_times),t)[1]
        if interevent_time <= 10:
            count +=1

    return(count/ len(target_times))



def time_agnostic(times1, times2):
    # calculate the time agnostic self entropy
    l1 = [(i,0) for i in times1]
    l2 = [(i,1) for i in times2]
    se = self_entropy_rate([e[1] for e in sorted(l1+l2, key = lambda x:x[0])])

    if se==0:
        return 0
    else:
        return 1/se