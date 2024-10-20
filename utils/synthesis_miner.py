import os
import sys
import time

import pandas as pd
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.algo.discovery.dfg.variants import native as dfg_inst
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.petri_utils import get_transition_by_name
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.objects.petri_net.utils import incidence_matrix
from pm4py.objects.petri_net.utils import check_soundness
from pm4py.objects.petri_net.utils.networkx_graph import create_networkx_directed_graph
from networkx.algorithms.components import is_strongly_connected
from pm4py.algo.analysis.woflan.algorithm import short_circuit_petri_net
from pm4py.objects.petri_net.utils import reachability_graph
from pm4py.algo.analysis.woflan import algorithm as woflan
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter
from pm4py.objects.petri_net.importer import importer as pnml_importer
from itertools import product
import networkx as nx
import numpy as np
import sympy
import re
import copy
import pm4py
from pm4py.objects.petri_net.utils.reduction import *
from pm4py.objects.petri_net import semantics

def get_bf_order(log, from_start=False, from_end=True, noise_threshold=0, exp=False):
    
    def get_init_order(log, uniq_a, uniq_a_count, where='start'):

        uniq_a_count_dict = uniq_a_count_dict = {uniq_a[i]:c for i, c in enumerate(uniq_a_count)}

        if where == 'start':
            init_activities = [[a['concept:name'] for a in trace][0] for trace in log]
        elif where == 'end':
            init_activities = [[a['concept:name'] for a in trace][-1] for trace in log]

        uniq_init_a, uniq_init_a_edge_weights = np.unique(init_activities, return_counts=True)
        init_nodes = np.array([(node_name, uniq_init_a_edge_weights[i], uniq_a_count_dict[node_name]) 
                               for i, node_name in enumerate(uniq_init_a)], 
                              dtype=[('node_name', 'object'), ('weight', 'int'), ('node_freq', 'int')])

        init_nodes_order_inds = np.flip(np.argsort(init_nodes, order=['weight', 'node_freq']))  
        ordered_init_nodes = init_nodes[init_nodes_order_inds]      
        ordered_init_nodes = [node[0] for node in ordered_init_nodes]

        return ordered_init_nodes
            
        
    def get_ordered_neighbor_nodes(activity_name:str, dfg, uniq_a, uniq_a_count, 
                                   direction='target'):
        
        
        uniq_a_ind_dict = {node:ind for ind, node in enumerate(uniq_a)}
        
        if direction == 'target':
            neighbor_nodes = np.array([(r[1], v, uniq_a_count[uniq_a_ind_dict[r[1]]]) 
                                     for r, v in dfg.items() if r[0] == activity_name and r[1] != activity_name], 
                                    dtype=[('node_name', 'object'), ('weight', 'int'), ('node_freq', 'int')])
            
        elif direction == 'source':
            neighbor_nodes = np.array([(r[0], v, uniq_a_count[uniq_a_ind_dict[r[0]]]) 
                                     for r, v in dfg.items() if r[1] == activity_name and r[0] != activity_name], 
                                    dtype=[('node_name', 'object'), ('weight', 'int'), ('node_freq', 'int')])            
        
        neighbor_nodes_order_inds = np.flip(np.argsort(neighbor_nodes, order=['weight', 'node_freq']))  
        ordered_neighbor_nodes = neighbor_nodes[neighbor_nodes_order_inds]      
        ordered_neighbor_nodes = [node[0] for node in ordered_neighbor_nodes]
                
        return ordered_neighbor_nodes
    
    def filter_dfg(dfg_dict, noise_threshold=0.2):
        
        # filter out going arcs
        threshold_dict = {a:max([v for k, v in dfg_dict.items() if k[0] == a])*noise_threshold 
                          for a in set([k[0] for k in dfg_dict])}
        filtered_dfg_dict = {k:v for k, v in dfg_dict.items() if v >= threshold_dict[k[0]]}

        # print(filtered_dfg_dict)
#         # filter incoming arcs
#         threshold_dict = {a:max([v for k, v in dfg_dict.items() if k[1] == a])*noise_threshold 
#                           for a in set([k[1] for k in dfg_dict])}

#         filtered_dfg_dict = {k:v for k, v in filtered_dfg_dict.items() if v >= threshold_dict[k[1]]}

        return filtered_dfg_dict

    
    unrolled_log = [a['concept:name'] for trace in log for a in trace]
    uniq_a, uniq_a_count = np.unique(unrolled_log, return_counts=True)
    uniq_a_sorted = np.flip(uniq_a[np.argsort(uniq_a_count)])
    n_a = len(uniq_a_sorted)

    dfg = dfg_inst.apply(log)
    
#     start_activities = set([[a['concept:name'] for a in trace][0] for trace in log])
#     end_activities = set([[a['concept:name'] for a in trace][-1] for trace in log])
    
    if from_start:

        # get order from start activities    
        order_from_start_a = []
        a_tb_explored = get_init_order(log, uniq_a, uniq_a_count, where='start')
        dfg_dict = {k:v for k, v in dfg.items()}
        dfg_dict = filter_dfg(dfg_dict, noise_threshold=noise_threshold)
#         print(dfg_dict)

        while len(a_tb_explored) != 0:

            ordered_target_nodes = get_ordered_neighbor_nodes(a_tb_explored[0], dfg_dict, uniq_a, uniq_a_count, 
                                                              direction='target')
            # filter out nodes that are already in a_tb_explored
            ordered_target_nodes = [n for n in ordered_target_nodes if n not in a_tb_explored]
            dfg_dict = {k:v for k, v in dfg_dict.items() if k[0] != a_tb_explored[0] and k[1] != a_tb_explored[0]}
            a_tb_explored = a_tb_explored + ordered_target_nodes
            order_from_start_a.append(a_tb_explored[0])
            a_tb_explored = a_tb_explored[1:]
    
    if from_end:

        # get order from start activities
        order_from_end_a = []
        a_tb_explored = get_init_order(log, uniq_a, uniq_a_count, where='end')
        dfg_dict = {k:v for k, v in dfg.items()}
        dfg_dict = filter_dfg(dfg_dict, noise_threshold=noise_threshold)
        # print(dfg_dict)
        
        while len(a_tb_explored) != 0:

            ordered_source_nodes = get_ordered_neighbor_nodes(a_tb_explored[0], dfg_dict, uniq_a, uniq_a_count, 
                                                              direction='source')
            # filter out nodes that are already in a_tb_explored
            ordered_source_nodes = [n for n in ordered_source_nodes if n not in a_tb_explored]
            dfg_dict = {k:v for k, v in dfg_dict.items() if k[0] != a_tb_explored[0] and k[1] != a_tb_explored[0]}
            a_tb_explored = a_tb_explored + ordered_source_nodes
            order_from_end_a.append(a_tb_explored[0])       
            a_tb_explored = a_tb_explored[1:]
            
    if exp:
        end_name = [a['concept:name'] for a in log[0]][-1]
            
    if from_start and not from_end:
        if exp:
            order_from_start_a.remove(end_name)
            order_from_start_a.insert(0,end_name)
        return order_from_start_a
    elif not from_start and from_end:
        if exp:
            order_from_end_a.remove(end_name)
            order_from_end_a.insert(0,end_name)
        return order_from_end_a
    elif from_start and from_end:
        
#         print(order_from_start_a)
#         print(order_from_end_a)
        order_from_both = []
        
        append_from_start = False
        for i in range(n_a):
            if append_from_start:
                a_tb_added = order_from_start_a[0]
            else:
                a_tb_added = order_from_end_a[0]
            
            append_from_start = not append_from_start
                
            order_from_both.append(a_tb_added)
            order_from_start_a.remove(a_tb_added)
            order_from_end_a.remove(a_tb_added)

            
        return order_from_both

def get_freq_order(log, exp=False):

    unrolled_log = [a['concept:name'] for trace in log for a in trace]
    uniq_a, uniq_a_count = np.unique(unrolled_log, return_counts=True)
    uniq_a_sorted = list(np.flip(uniq_a[np.argsort(uniq_a_count)]))
    if exp:
        end_name = [a['concept:name'] for a in log[0]][-1]
        uniq_a_sorted.remove(end_name)
        uniq_a_sorted.insert(0,end_name)
    return uniq_a_sorted

def get_dfs_order(log, from_end=True):
    
    def get_init_order(log, uniq_a, uniq_a_count, where='start'):

        uniq_a_count_dict = uniq_a_count_dict = {uniq_a[i]:c for i, c in enumerate(uniq_a_count)}

        if where == 'start':
            init_activities = [[a['concept:name'] for a in trace][0] for trace in log]
        elif where == 'end':
            init_activities = [[a['concept:name'] for a in trace][-1] for trace in log]

        uniq_init_a, uniq_init_a_edge_weights = np.unique(init_activities, return_counts=True)
        init_nodes = np.array([(node_name, uniq_init_a_edge_weights[i], uniq_a_count_dict[node_name]) 
                               for i, node_name in enumerate(uniq_init_a)], 
                              dtype=[('node_name', 'object'), ('weight', 'int'), ('node_freq', 'int')])

        init_nodes_order_inds = np.flip(np.argsort(init_nodes, order=['weight', 'node_freq']))  
        ordered_init_nodes = init_nodes[init_nodes_order_inds]      
        ordered_init_nodes = [node[0] for node in ordered_init_nodes]

        return ordered_init_nodes
            
        
    def get_ordered_neighbor_nodes(activity_name:str, dfg, uniq_a, uniq_a_count, 
                                   direction='target'):
        
        
        uniq_a_ind_dict = {node:ind for ind, node in enumerate(uniq_a)}
        
        if direction == 'target':
            neighbor_nodes = np.array([(r[1], v, uniq_a_count[uniq_a_ind_dict[r[1]]]) 
                                     for r, v in dfg.items() if r[0] == activity_name and r[1] != activity_name], 
                                    dtype=[('node_name', 'object'), ('weight', 'int'), ('node_freq', 'int')])
            
        elif direction == 'source':
            neighbor_nodes = np.array([(r[0], v, uniq_a_count[uniq_a_ind_dict[r[0]]]) 
                                     for r, v in dfg.items() if r[1] == activity_name and r[0] != activity_name], 
                                    dtype=[('node_name', 'object'), ('weight', 'int'), ('node_freq', 'int')])            
        
        neighbor_nodes_order_inds = np.flip(np.argsort(neighbor_nodes, order=['weight', 'node_freq']))  
        ordered_neighbor_nodes = neighbor_nodes[neighbor_nodes_order_inds]      
        ordered_neighbor_nodes = [node[0] for node in ordered_neighbor_nodes]
                
        return ordered_neighbor_nodes
    
    def filter_dfg(dfg_dict, noise_threshold=0.2):
        
        # filter out going arcs
        threshold_dict = {a:max([v for k, v in dfg_dict.items() if k[0] == a])*noise_threshold
                          for a in set([k[0] for k in dfg_dict])}
        filtered_dfg_dict = {k:v for k, v in dfg_dict.items() if v >= threshold_dict[k[0]]}

        # print(filtered_dfg_dict)
#         # filter incoming arcs
#         threshold_dict = {a:max([v for k, v in dfg_dict.items() if k[1] == a])*noise_threshold 
#                           for a in set([k[1] for k in dfg_dict])}

#         filtered_dfg_dict = {k:v for k, v in filtered_dfg_dict.items() if v >= threshold_dict[k[1]]}

        return filtered_dfg_dict

    
    unrolled_log = [a['concept:name'] for trace in log for a in trace]
    uniq_a, uniq_a_count = np.unique(unrolled_log, return_counts=True)
    uniq_a_sorted = np.flip(uniq_a[np.argsort(uniq_a_count)])
    n_a = len(uniq_a_sorted)

    dfg = dfg_inst.apply(log)
    

    if from_end:

        # get order from end activities
        order_from_end_a = []
        a_tb_explored = get_init_order(log, uniq_a, uniq_a_count, where='end')
        dfg_dict = {k:v for k, v in dfg.items()}
        # dfg_dict = filter_dfg(dfg_dict, noise_threshold=noise_threshold)

        gamma = [a_tb_explored[0]]
        stack = a_tb_explored.copy()[1:]
        A = set(uniq_a)

        while len(A) != len(gamma):
            A_prime = A.difference(set(gamma))
            gamma_prime = get_ordered_neighbor_nodes(gamma[-1], dfg_dict, uniq_a, uniq_a_count, direction='source')
            gamma_prime = [a for a in gamma_prime if a in A_prime]

            if len(gamma_prime) == 0:
                gamma.append(stack[0])
            else:
                gamma.append(gamma_prime[0])
            
            gamma_prime = [a for a in gamma_prime[1:] if a not in stack]
            stack = gamma_prime + stack

            stack = [s for s in stack if s in A.difference(set(gamma))]

    else:

        # get order from start activities
        order_from_start_a = []
        a_tb_explored = get_init_order(log, uniq_a, uniq_a_count, where='start')
        dfg_dict = {k:v for k, v in dfg.items()}
        # dfg_dict = filter_dfg(dfg_dict, noise_threshold=noise_threshold)

        gamma = [a_tb_explored[0]]
        stack = a_tb_explored.copy()[1:]
        A = set(uniq_a)

        while len(A) != len(gamma):
            A_prime = A.difference(set(gamma))
            gamma_prime = get_ordered_neighbor_nodes(gamma[-1], dfg_dict, uniq_a, uniq_a_count, direction='target')
            gamma_prime = [a for a in gamma_prime if a in A_prime]

            if len(gamma_prime) == 0:
                gamma.append(stack[0])
            else:
                gamma.append(gamma_prime[0])
            
            gamma_prime = [a for a in gamma_prime[1:] if a not in stack]
            stack = gamma_prime + stack

            stack = [s for s in stack if s in A.difference(set(gamma))]
        
        end_name = [a['concept:name'] for a in log[0]][-1]
        gamma.remove(end_name)
        gamma.insert(0,end_name)
    
    return gamma

def remove_tran_by_name(net, trans_name='short_circuited_transition'):
    """
    Remove a transition from a Petri net by name, default name: short_circuited_transition

    Parameters
    ----------
    net
        Petri net
    trans_name
        the name of the transition to be removed

    Returns
    ----------
    net
        Petri net
    """

    new_net = copy.deepcopy(net)
    
    trans = get_transition_by_name(new_net, trans_name)
    if trans in new_net.transitions:
        in_arcs = trans.in_arcs
        for arc in in_arcs:
            place = arc.source
            place.out_arcs.remove(arc)
            new_net.arcs.remove(arc)
        out_arcs = trans.out_arcs
        for arc in out_arcs:
            place = arc.target
            place.in_arcs.remove(arc)
            new_net.arcs.remove(arc)
        new_net.transitions.remove(trans)
    return new_net


def initialize_net(first_t_label:str, optional=False, loop=False, short_circuit=True):
    net = PetriNet("new_petri_net")
    
    if (not optional) and (not loop):
        labels = ['start_tau', first_t_label, 'end_tau']
    else:
        labels = ['start_tau', first_t_label, 'tau_0', 'end_tau']
    
    if short_circuit:
        labels.append('short_circuited_transition')
    
    label_transition_dict = {}

    for i in range(0, len(labels)):
        if 'tau' in labels[i]:
            label_transition_dict[labels[i]] = PetriNet.Transition(labels[i], None)
        else:
            label_transition_dict[labels[i]] = PetriNet.Transition(labels[i], labels[i])
        net.transitions.add(label_transition_dict[labels[i]])
        
        
    place_dict = {}

    place_names = ['source', 'p0', 'p1', 'sink']

    for i in range(0, len(place_names)):
        place_dict[place_names[i]] = PetriNet.Place(place_names[i])
        net.places.add(place_dict[place_names[i]])
        
    add_arc_from_to(place_dict['source'], label_transition_dict['start_tau'], net)
    add_arc_from_to(label_transition_dict['start_tau'], place_dict['p0'], net)
        
    if not (optional and loop):
        
        # (not optional) and (not loop)
        add_arc_from_to(place_dict['p0'], label_transition_dict[first_t_label], net)
        add_arc_from_to(label_transition_dict[first_t_label], place_dict['p1'], net)

        if optional and (not loop):
            add_arc_from_to(place_dict['p0'], label_transition_dict['tau_0'], net)
            add_arc_from_to(label_transition_dict['tau_0'], place_dict['p1'], net)

        elif (not optional) and loop:
            add_arc_from_to(place_dict['p1'], label_transition_dict['tau_0'], net)
            add_arc_from_to(label_transition_dict['tau_0'], place_dict['p0'], net)

    else: # optional and loop
        add_arc_from_to(place_dict['p0'], label_transition_dict['tau_0'], net)
        add_arc_from_to(label_transition_dict['tau_0'], place_dict['p1'], net)
        add_arc_from_to(place_dict['p1'], label_transition_dict[first_t_label], net)
        add_arc_from_to(label_transition_dict[first_t_label], place_dict['p0'], net)
        
    add_arc_from_to(place_dict['p1'], label_transition_dict['end_tau'], net)
    add_arc_from_to(label_transition_dict['end_tau'], place_dict['sink'], net)
    
    if short_circuit:
        add_arc_from_to(place_dict['sink'], label_transition_dict['short_circuited_transition'], net)
        add_arc_from_to(label_transition_dict['short_circuited_transition'], place_dict['source'], net)
        
    im = Marking({place_dict['source']: 1})
    fm = Marking({place_dict['sink']: 1})
    
    return net, im, fm

def create_pn_from_incidence_mat(incidence_mat, places_dict=None, trans_dict=None, return_net_dict=True):
    '''
        parameters
            incidence_mat: incidence matrix
            trans_dict: a dictionary mapping the name of the transition to the colmn index in the incidence_mat
            places_dict: a dictionary mapping the name of the place to the row index in the incidence_mat

        return
            a petri net
    '''
    net = PetriNet("new_petri_net")
    
    if (np.sum(np.all(incidence_mat <= 0, axis=1)) == 1) and (np.sum(np.all(incidence_mat >= 0, axis=1)) == 1):
        is_workflow_net = True
    else:
        is_workflow_net = False
    
    # create transitions
    if trans_dict == None:
        trans_dict = {'t{}'.format(i):i for i in range(incidence_mat.shape[1])}
    label_trans_dict = {}
    trans_labels = list(trans_dict.keys())
    for i in range(0, len(trans_labels)):
        if 'tau' in trans_labels[i]:
            label_trans_dict[trans_labels[i]] = PetriNet.Transition(trans_labels[i], None)
        else:
            label_trans_dict[trans_labels[i]] = PetriNet.Transition(trans_labels[i], trans_labels[i])
        net.transitions.add(label_trans_dict[trans_labels[i]])
        
    
    # create places
    if places_dict == None:
        places_dict = {'p{}'.format(i):i for i in range(incidence_mat.shape[0])}
    place_dict = {}
    place_names = list(places_dict.keys())
    for i in range(0, len(place_names)):
        place_dict[place_names[i]] = PetriNet.Place(place_names[i])
        net.places.add(place_dict[place_names[i]])
        
    # connect nodes using incidence_mat
    for c in product(places_dict.keys(), trans_dict.keys()):
        place_name = c[0]
        trans_label = c[1]
        value_in_mat = incidence_mat[places_dict[place_name], trans_dict[trans_label]]
        
        if value_in_mat == -1:
            add_arc_from_to(place_dict[place_name], label_trans_dict[trans_label], net)
        elif value_in_mat == 1:
            add_arc_from_to(label_trans_dict[trans_label], place_dict[place_name], net)

#     if 'source' in place_dict.keys() and 'sink' in place_dict.keys():
#         im = Marking({place_dict['source']: 1})
#         fm = Marking({place_dict['sink']: 1})

    if return_net_dict:
        net_dict = {}
        # since the incidence_matrix.construct() sort the nodes by name
        m = incidence_matrix.construct(net)
        mat = np.array(m.a_matrix)
        places_dict = {k.name: int(v) for k, v in m.places.items()}
        trans_dict = {k.name: int(v) for k, v in m.transitions.items()}
        net_dict['petri'] = net
        net_dict['incidence_mat'] = mat
        net_dict['places_dict'] = places_dict
        net_dict['trans_dict'] = trans_dict
        
        if is_workflow_net:
            # find source_ind and sink_ind in mat
            source_ind = np.where(np.all(net_dict['incidence_mat'] <= 0, axis=1))[0][0]
            sink_ind = np.where(np.all(net_dict['incidence_mat'] >= 0, axis=1))[0][0]
            source_name = [k for k, v in places_dict.items() if v == source_ind][0]
            sink_name = [k for k, v in places_dict.items() if v == sink_ind][0]
            for p in net.places:
                if p.name == source_name:
                    p_source = p
                    im = Marking({p_source: 1})
                elif p.name == sink_name:
                    p_sink = p
                    fm = Marking({p_sink: 1})
                    
            net_dict['im'] = im
            net_dict['fm'] = fm
        
        elif 'short_circuited_transition' in trans_dict.keys():
            break_sc_petri = remove_tran_by_name(net_dict['petri'], trans_name='short_circuited_transition')
            net_dict['break_sc_petri'] = break_sc_petri
            for p in break_sc_petri.places:
                if p.name == 'source':
                    p_source = p
                    im = Marking({p_source: 1})
                elif p.name == 'sink':
                    p_sink = p
                    fm = Marking({p_sink: 1})

            net_dict['im'] = im
            net_dict['fm'] = fm
            
        
        return net_dict
    else:
        return net

def is_fc_net_from_mat(incidence_mat):
    
    n_trans = incidence_mat.shape[1]
    t_comb = combinations(range(n_trans), 2)
    
    b_list = []
    for c in list(t_comb):
        in_nodes_t1 = set([i for i, b in enumerate(incidence_mat[:,c[0]] == -1) if b])
        in_nodes_t2 = set([i for i, b in enumerate(incidence_mat[:,c[1]] == -1) if b])

        is_in_nodes_same = in_nodes_t1 == in_nodes_t2
        is_in_nodes_disjoint = in_nodes_t1.intersection(in_nodes_t2) == set()

        b_list.append(is_in_nodes_same or is_in_nodes_disjoint)

    return all(b_list)


def get_all_combs(l:set):

    all_combinations = []

    for r in range(1, len(l) + 1):

        combs = combinations(l, r)

        combinations_list = list(map(set, combs))

        all_combinations += combinations_list

    return all_combinations

def get_all_combs_gen(l:set):

    all_combinations = [combinations(l, r) for r in range(1, len(l) + 1)]

#     for r in range(1, len(l) + 1):

#         combs = combinations(l, r)

#         combinations_list = list(map(set, combs))

#         all_combinations += combinations_list
    all_combs_gen = map(set, chain(*all_combinations))

    return all_combs_gen

def check_every_siphon_has_source(incidence_mat, source_ind, check_new_p_only=True):
    
    # print("check_every_siphon_has_source")
    
    def is_siphon(incidence_mat, places_ind_set):

        pre_set = set((incidence_mat[list(places_ind_set),] == 1).any(axis=0).nonzero()[0])
        post_set = set((incidence_mat[list(places_ind_set),] == -1).any(axis=0).nonzero()[0])
        
#         print(pre_set, post_set)
        
        return pre_set.issubset(post_set)
    
    all_place_combs_gen = get_all_combs_gen([i for i in range(incidence_mat.shape[0])])
    
#     is_siphon_b = [is_siphon(incidence_mat=incidence_mat, places_ind_set=places_ind_set) 
#                     for places_ind_set in all_place_combs_gen]
    
#     siphons = [all_place_combs[i] for i, b in enumerate(is_siphon_b) if b]
    
    # only cosider the places_ind_set which has new_p
    if check_new_p_only:
        
        all_place_combs_gen = (places_ind_set for places_ind_set in all_place_combs_gen 
                               if (incidence_mat.shape[0] - 1) in places_ind_set)
    
    siphons_gen = (places_ind_set for places_ind_set in all_place_combs_gen 
                   if is_siphon(incidence_mat=incidence_mat, places_ind_set=places_ind_set))
    
    siphons_have_source_b_gen = (source_ind in siphon for siphon in siphons_gen)
    
    return all(siphons_have_source_b_gen)


def check_easy_soundness_of_wfnet_from_mat(incidence_mat):
    # print("check_easy_soundness")
    net = create_pn_from_incidence_mat(incidence_mat, places_dict=None, trans_dict=None, return_net_dict=False)
#     gviz = pn_visualizer.apply(net)
#     pn_visualizer.view(gviz)
    return check_soundness.check_easy_soundness_of_wfnet(net)


def is_sound_from_mat_by_woflan(incidence_mat):
    # print("check_sound_by_woflan")
    net_dict = create_pn_from_incidence_mat(incidence_mat=incidence_mat, return_net_dict=True)
    is_sound = woflan.apply(net=net_dict['petri'], 
                             i_m=net_dict['im'], 
                             f_m=net_dict['fm'], 
                             parameters={woflan.Parameters.RETURN_ASAP_WHEN_NOT_SOUND: True,
                                            woflan.Parameters.PRINT_DIAGNOSTICS: False,
                                            woflan.Parameters.RETURN_DIAGNOSTICS: False})
    return is_sound

## generator
def get_new_nodes_gen_by_LD_rules(incidence_mat, source_ind:int, sink_ind:int, 
                                  start_tau_ind:int, end_tau_ind:int, short_circuit_ind: int, 
                                  node_type='t', tbc_nodes_inds_dict=None, check_sound=True):
    
    # print('get_new_nodes_gen_by_LD_rules')
    
    def is_all_pos_or_neg_vec(cand_vector):
        return (all(cand_vector >= 0) or all(cand_vector <= 0))
    
    def is_LD(cand_vector, node_type, incidence_mat, short_circuit_ind):
        
        if node_type == 't':
            # print('checking_LD_t')
            new_mat = np.concatenate((incidence_mat, cand_vector.reshape(-1, 1)), axis=1)
            if is_fc_net_from_mat(new_mat):
                _, inds = sympy.Matrix(new_mat).rref()
                include_in_final = (new_mat.shape[1] - 1) not in inds
            else:
                include_in_final = False
        elif node_type == 'p':
            # print('checking_LD_p')
            new_mat = np.concatenate((incidence_mat, cand_vector.reshape(1, -1)), axis=0)
            if check_sound:
                # mat_tb_checked_sound = np.delete(new_mat, obj=short_circuit_ind, axis=1)
                # if is_fc_net_from_mat(new_mat) and is_sound_from_mat_by_woflan(mat_tb_checked_sound):
                if is_fc_net_from_mat(new_mat):
                    _, inds = sympy.Matrix(new_mat).T.rref()
                    include_in_final = ((new_mat.shape[0] - 1) not in inds)
#                 if is_fc_net_from_mat(new_mat):
#                     _, inds = sympy.Matrix(new_mat).T.rref()
#                     include_in_final = ((new_mat.shape[0] - 1) not in inds) and check_easy_soundness_of_wfnet_from_mat(mat_tb_checked_sound)
                else:
                    include_in_final = False                
            else:         
                if is_fc_net_from_mat(new_mat) and check_every_siphon_has_source(new_mat, source_ind):
                    _, inds = sympy.Matrix(new_mat).T.rref()
                    include_in_final = (new_mat.shape[0] - 1) not in inds
                else:
                    include_in_final = False
                
        return include_in_final
    
    n_places, n_trans = incidence_mat.shape
    
    # create initial candidate vectors considering special nodes in worflow net: 
    # e.g. start, short_circuit transitions, source & sink places, etc.
    if node_type == 't':
        prod_arg = [[1, 0, -1] for i in range(n_places)]
        prod_arg[source_ind], prod_arg[sink_ind] = [0], [0]
        
        # if there exists an indices of nodes that can be used to reduce the #possible new nodes.
        if tbc_nodes_inds_dict: 
            for p_ind in range(n_places):
                if p_ind not in tbc_nodes_inds_dict['places']:
                    prod_arg[p_ind] = [0]
                    
        cand_vectors = product(*prod_arg)
    elif node_type == 'p':
        prod_arg = [[1, 0, -1] for i in range(n_trans)]
        prod_arg[start_tau_ind] = [1, 0]
        prod_arg[end_tau_ind] = [-1, 0]
        prod_arg[short_circuit_ind] = [0]
        
        # if there exists an indices of nodes that can be used to reduce the #possible new nodes.
        if tbc_nodes_inds_dict: 
            for t_ind in range(n_trans):
                if t_ind not in tbc_nodes_inds_dict['transitions']:
                    prod_arg[t_ind] = [0]
                elif t_ind in tbc_nodes_inds_dict['pre']:
                    prod_arg[t_ind] = [1]
                elif t_ind in tbc_nodes_inds_dict['post']:
                    prod_arg[t_ind] = [-1]

        cand_vectors = product(*prod_arg)
        
        # filter candidate places that are just the reverse, e.g. [1, 0, -1] => [-1, 0, 1]
        cand_vectors = (cand_vec for cand_vec in cand_vectors if cand_vec not in (-incidence_mat).tolist())
    
    # filter out vectors where all entries >= 0 or <= 0
    cand_vectors_filtered = (cand_vector for cand_vector in cand_vectors 
                             if not is_all_pos_or_neg_vec(np.array(cand_vector)))
    

    # print('len_cand_vectors_filtered', len(list((cand_vector for cand_vector in cand_vectors if not is_all_pos_or_neg_vec(np.array(cand_vector))))))
#     final_cand_vectors_bool = []
    
#     # check linear dependence
#     for cand_vec in cand_vectors_filtered:
#         if node_type == 't':
#             new_mat = np.concatenate((incidence_mat, cand_vec.reshape(-1, 1)), axis=1)
#             if is_fc_net_from_mat(new_mat):
#                 _, inds = sympy.Matrix(new_mat).rref()
#                 include_in_final = (new_mat.shape[1] - 1) not in inds
#             else:
#                 include_in_final = False
#         elif node_type == 'p':
#             new_mat = np.concatenate((incidence_mat, cand_vec.reshape(1, -1)), axis=0)
#             if is_fc_net_from_mat(new_mat) and check_every_siphon_has_source(new_mat, source_ind):
#                 _, inds = sympy.Matrix(new_mat).T.rref()
#                 include_in_final = (new_mat.shape[0] - 1) not in inds
#             else:
#                 include_in_final = False
        
# #         print(new_mat.shape, inds)
#         final_cand_vectors_bool.append(include_in_final)
    
#     print(np.sum(final_cand_vectors_bool))
    final_cand_vectors_gen = (np.array(cand_vector) for cand_vector in cand_vectors_filtered 
                              if is_LD(cand_vector=np.array(cand_vector), node_type=node_type, 
                                       incidence_mat=incidence_mat, short_circuit_ind=short_circuit_ind))
    
#     print("[1,  0,  0, -1,  0,  0,  1, -1] in final_cand_vectors")
#     print([1,  0,  0, -1,  0,  0,  1, -1] in final_cand_vectors.tolist())
    
    return final_cand_vectors_gen

def apply_LD_t_rule(incidence_mat, new_t_name:str, new_t_col, trans_dict:dict):
    '''
        parameters
            incidence_mat
            new_t_name: the to be added transition's name
            new_t_col: the column for the new transition
            trans_dict
        return 
            updated_incidence_mat: an updated incidence matrix after applying the transition linear dependence rule
            updated_trans_dict
    '''
    new_t_col = np.array(new_t_col)
    updated_incidence_mat = np.concatenate((incidence_mat, new_t_col.reshape(-1, 1)), axis=1)
    
    updated_trans_dict = trans_dict.copy()
    updated_trans_dict[new_t_name] = updated_incidence_mat.shape[1] - 1
    
    return updated_incidence_mat, updated_trans_dict

def apply_LD_p_rule(incidence_mat, new_p_row, places_dict:dict):
    '''
        parameters
            incidence_mat
            new_p_row: the row for the new place
            places_dict
        return 
            updated_incidence_mat: an updated incidence matrix after applying the place linear dependence rule
            updated_places_dict
    '''
    
    new_p_row = np.array(new_p_row)
    updated_incidence_mat = np.concatenate((incidence_mat, new_p_row.reshape(1, -1)), axis=0)
    
    updated_places_dict = places_dict.copy()
    # get a name for the new place
    p_num_nested_l = [re.findall("\d+", p_name) for p_name in places_dict.keys()]
    p_num_l = [int(p_num) for sublist in p_num_nested_l for p_num in sublist]
    
    if len(p_num_l) == 0:
        new_p_name = 'p0'
    else:
        new_p_name = 'p{}'.format(max(p_num_l) + 1)

    updated_places_dict[new_p_name] = updated_incidence_mat.shape[0] - 1
    
    return updated_incidence_mat, updated_places_dict

def check_abstration_applicable(incidence_mat, R: set, S: set):
    is_applicable = np.all([incidence_mat[ind[1], ind[0]] == 1 for ind in product(R, S)])
    return is_applicable

def get_possible_abs_pairs_gen(incidence_mat, source_ind:int, sink_ind:int, tbc_nodes_inds_dict=None):
    
    # print("get_possible_abs_pairs_gen")
    
    if tbc_nodes_inds_dict:
        R = [node_ind for node_ind in tbc_nodes_inds_dict['transitions'] if node_ind not in tbc_nodes_inds_dict['post']]
        S = tbc_nodes_inds_dict['places']
    else:    
        R = [t_i for t_i in range(incidence_mat.shape[1])]
        S = [p_i for p_i in range(incidence_mat.shape[0]) if p_i not in [source_ind, sink_ind]]
    R_combs = get_all_combs_gen(R)
    S_combs = get_all_combs_gen(S)

    all_products_gen = product(R_combs, S_combs)
    
    possible_abs_pairs_gen = (c for c in all_products_gen if 
                              check_abstration_applicable(incidence_mat, set(c[0]), set(c[1])))
    
    return possible_abs_pairs_gen

def apply_abstraction_rule(incidence_mat, new_t_name:str, R:set, S:set,
                           places_dict:dict, trans_dict:dict, new_t_skippable=False, new_t_in_loop=False):
    '''
        parameters
            incidence_mat
            new_t_name: the to be added transition's name
            R: a set of transitions index R
            S: a set of places index S
            trans_dict
            places_dict
            new_t_skippable: whether to make t skippable
        return
            updated_incidence_mat: an updated incidence matrix after applying the abstratoin rule
            updated_trans_dict
            updated_places_dict
    '''
    n_places, n_trans = incidence_mat.shape
    updated_incidence_mat = incidence_mat.copy()
    # test if applicable
    is_applicable = np.all([incidence_mat[ind[1], ind[0]] == 1 for ind in product(R, S)])

    if not is_applicable:
        raise Exception('Abstraction rule is not applicable')
    else:

        updated_trans_dict = trans_dict.copy()
        updated_places_dict = places_dict.copy()

        # get a name for the new place
        p_num_nested_l = [re.findall("\d+", p_name) for p_name in places_dict.keys()]
        p_num_l = [int(p_num) for sublist in p_num_nested_l for p_num in sublist]
        if len(p_num_l) == 0:
            new_p_name = 'p0'
        else:
            new_p_name = 'p{}'.format(max(p_num_l) + 1)

        # break the connection from R to S in incidence_mat
        for ind in product(R, S):
                updated_incidence_mat[ind[1], ind[0]] = 0

        # connect R to new p
        new_p_row = np.array([v in R for v in range(n_trans)]).astype(int)
        updated_incidence_mat = np.concatenate((updated_incidence_mat, new_p_row.reshape(1, -1)), axis=0)

        # connect new t to S
        new_t_col = np.array([v in S for v in range(updated_incidence_mat.shape[0])]).astype(int)
        new_t_col[-1] = -1
        updated_incidence_mat = np.concatenate((updated_incidence_mat, new_t_col.reshape(-1, 1)), axis=1)

        updated_places_dict[new_p_name] = updated_incidence_mat.shape[0] - 1

        if new_t_name == None:
            # get a name for the new tau
            tau_num_nested_l = [re.findall("(?!tau)\d+", tau_name) for tau_name in trans_dict.keys()]
            tau_num_l = [int(tau_num) for sublist in tau_num_nested_l for tau_num in sublist]

            if len(tau_num_l) == 0:
                new_tau_name = 'tau_0'
            else:
                new_tau_name = 'tau_{}'.format(max(tau_num_l) + 1)

            updated_trans_dict[new_tau_name] = updated_incidence_mat.shape[1] - 1
        else:
            updated_trans_dict[new_t_name] = updated_incidence_mat.shape[1] - 1
        
#         # make the new t skipable
#         if new_t_skippable or new_t_in_loop:
            
#             if new_t_skippable and new_t_in_loop: # t in a skippable loop
#                 # reverse t's connection
#                 updated_incidence_mat[:,-1] = -updated_incidence_mat[:,-1]
#                 updated_incidence_mat = np.concatenate((updated_incidence_mat, 
#                                         -updated_incidence_mat[:,-1].reshape(-1, 1)), axis=1)
#             if new_t_skippable and (not new_t_in_loop):
#                 updated_incidence_mat = np.concatenate((updated_incidence_mat, 
#                                                         updated_incidence_mat[:,-1].reshape(-1, 1)), axis=1)
#             elif (not new_t_skippable) and new_t_in_loop:
#                 updated_incidence_mat = np.concatenate((updated_incidence_mat, 
#                                                         -updated_incidence_mat[:,-1].reshape(-1, 1)), axis=1)
                
#             # get a name for the new tau
#             tau_num_nested_l = [re.findall("(?!tau)\d+", tau_name) for tau_name in trans_dict.keys()]
#             tau_num_l = [int(tau_num) for sublist in tau_num_nested_l for tau_num in sublist]

#             if len(tau_num_l) == 0:
#                 new_tau_name = 'tau_0'
#             else:
#                 new_tau_name = 'tau_{}'.format(max(tau_num_l) + 1)

#             updated_trans_dict[new_tau_name] = updated_incidence_mat.shape[1] - 1

        if new_t_skippable or new_t_in_loop:
            
            if new_t_in_loop:                
                # build a new with skippable == False, loop == True first
                # if skippable == True, loop == True, swap the new_t_name, and new tau later
                new_R = set([updated_trans_dict[new_t_name]])
                new_S = set(np.where(updated_incidence_mat[:,-1] == 1)[0])
                updated_incidence_mat, updated_places_dict, updated_trans_dict = apply_abstraction_rule(
                                                                                    incidence_mat=updated_incidence_mat, 
                                                                                    new_t_name=None,
                                                                                    R=new_R,
                                                                                    S=new_S,
                                                                                    places_dict=updated_places_dict, 
                                                                                    trans_dict=updated_trans_dict, 
                                                                                    new_t_skippable=False, 
                                                                                    new_t_in_loop=False)
                updated_incidence_mat = np.concatenate((updated_incidence_mat, 
                                                        -updated_incidence_mat[:,updated_trans_dict[new_t_name]].reshape(-1, 1)), axis=1)
                
            else: # skippable == True, loop == False
                updated_incidence_mat = np.concatenate((updated_incidence_mat, 
                                                        updated_incidence_mat[:,-1].reshape(-1, 1)), axis=1)
                
            # get a name for the new tau
            tau_num_nested_l = [re.findall("(?!tau)\d+", tau_name) for tau_name in updated_trans_dict.keys()]
            tau_num_l = [int(tau_num) for sublist in tau_num_nested_l for tau_num in sublist]

            if len(tau_num_l) == 0:
                new_tau_name = 'tau_0'
            else:
                new_tau_name = 'tau_{}'.format(max(tau_num_l) + 1)
            
            if new_t_skippable and new_t_in_loop: # swap the new_t_name and new tau in the updated_trans_dict
                updated_trans_dict[new_tau_name] = updated_trans_dict[new_t_name]
                updated_trans_dict[new_t_name] = updated_incidence_mat.shape[1] - 1
            else:
                updated_trans_dict[new_tau_name] = updated_incidence_mat.shape[1] - 1
        
        return updated_incidence_mat, updated_places_dict, updated_trans_dict


def apply_LD_p_then_abs_rules(incidence_mat, new_t_name:str, new_p_row, 
                              places_dict:dict, trans_dict:dict, new_t_skippable=False, new_t_in_loop=False):
    '''
        parameters
            incidence_mat
            new_t_name: the to be added transition's name
            trans_dict
            new_t_skippable=False
            new_t_in_loop=False
        return 
            updated_incidence_mat: an updated incidence matrix after applying the transition linear dependence rule
            updated_trans_dict
    '''
    
    new_p_row = np.array(new_p_row)
    updated_incidence_mat = np.concatenate((incidence_mat, new_p_row.reshape(1, -1)), axis=0)
    
    # get index for R: a set of transitions index R
    new_p_preset = set(np.where(new_p_row == 1)[0])
    new_p_index = set([updated_incidence_mat.shape[0] - 1])
    
    updated_places_dict = places_dict.copy()
    # get a name for the new place
    p_num_nested_l = [re.findall("\d+", p_name) for p_name in places_dict.keys()]
    p_num_l = [int(p_num) for sublist in p_num_nested_l for p_num in sublist]
    
#     print(updated_incidence_mat, new_p_preset, new_p_index)
    
    if len(p_num_l) == 0:
        new_p_name = 'p0'
    else:
        new_p_name = 'p{}'.format(max(p_num_l) + 1)

    updated_places_dict[new_p_name] = updated_incidence_mat.shape[0] - 1
    
    final_incidence_mat, final_places_dict, final_trans_dict = apply_abstraction_rule(incidence_mat=updated_incidence_mat, 
                                                                                      new_t_name=new_t_name, 
                                                                                      R=new_p_preset, 
                                                                                      S=new_p_index, 
                                                                                      trans_dict=trans_dict, 
                                                                                      places_dict=updated_places_dict, 
                                                                                      new_t_skippable=new_t_skippable, 
                                                                                      new_t_in_loop=new_t_in_loop)
    
    return final_incidence_mat, final_places_dict, final_trans_dict


# updated
def get_all_new_rule_pairs(incidence_mat, source_ind:int, sink_ind:int, ignore_ss=True, tbc_nodes_inds_dict=None):
        
    # get all clusters
    clusters = []
    n_places = incidence_mat.shape[0]
    if tbc_nodes_inds_dict:
        places_inds = tbc_nodes_inds_dict['places']
    else:  
        if ignore_ss:
            places_inds = [i for i in range(n_places) if i not in [source_ind, sink_ind]]
        else:    
            places_inds = [i for i in range(n_places)]

    for i in places_inds:

        unrolled_p = [p_i for c in clusters for p_i in c[0]]
        if i not in unrolled_p:
            c_p = [i]
            c_t = []
            all_t_included = np.all([t_i in c_t for t_i in np.nonzero(incidence_mat[c_p, ] == -1)[1]])
            all_p_included = np.all([p_i in c_p for p_i in np.nonzero(incidence_mat[:,c_t] == -1)[0]])
            while not (all_p_included and all_t_included):

                for t_i in np.unique(np.nonzero(incidence_mat[c_p, ] == -1)[1]):
                    if t_i not in c_t:
                        c_t.append(t_i)

                for p_i in np.unique(np.nonzero(incidence_mat[:,c_t] == -1)[0]):
                    if p_i not in c_p:
                        c_p.append(p_i)

                all_t_included = np.all([t_i in c_t for t_i in np.nonzero(incidence_mat[c_p, ] == -1)[1]])
                all_p_included = np.all([p_i in c_p for p_i in np.nonzero(incidence_mat[:,c_t] == -1)[0]])

            clusters.append([c_p, c_t])
            
    possible_pairs = []
    for SR_C_pair in clusters:

        S_C = SR_C_pair[0]
        R_C = SR_C_pair[1]

        for pair in product(get_all_combs_gen(S_C), get_all_combs_gen(R_C)):
            S = pair[0]
            R = pair[1]
            if (S == set(S_C)) or (R == set(R_C)):
                possible_pairs.append(list(pair))        
        
    return possible_pairs

def apply_new_rule(incidence_mat, new_t_name:str, R:set, S:set, 
                           places_dict:dict, trans_dict:dict, new_t_skippable=False, new_t_in_loop=False):
    '''
        parameters
            incidence_mat
            new_t_name: the to be added transition's name
            R: a set of transitions index R
            S: a set of places index S
            trans_dict
            places_dict
            new_t_skippable: whether to make t skippable
        return 
            updated_incidence_mat: an updated incidence matrix after applying the abstratoin rule
            updated_trans_dict
            updated_places_dict
    '''
    n_places, n_trans = incidence_mat.shape
    updated_incidence_mat = incidence_mat.copy()
    
    updated_trans_dict = trans_dict.copy()
    updated_places_dict = places_dict.copy()

    # get a name for the new place
    p_num_nested_l = [re.findall("\d+", p_name) for p_name in places_dict.keys()]
    p_num_l = [int(p_num) for sublist in p_num_nested_l for p_num in sublist]
    if len(p_num_l) == 0:
        new_p_name = 'p0'
    else:
        new_p_name = 'p{}'.format(max(p_num_l) + 1)

    # break the connection from S to R in incidence_mat
    for ind in product(S, R):
            updated_incidence_mat[ind[0], ind[1]] = 0

    # connect S to new t
    new_t_col = -np.array([v in S for v in range(updated_incidence_mat.shape[0])]).astype(int)
    updated_incidence_mat = np.concatenate((updated_incidence_mat, new_t_col.reshape(-1, 1)), axis=1)


    # connect new p to R
    new_p_row = -np.array([v in R for v in range(updated_incidence_mat.shape[1])]).astype(int)
    new_p_row[-1] = 1
    updated_incidence_mat = np.concatenate((updated_incidence_mat, new_p_row.reshape(1, -1)), axis=0)

    updated_places_dict[new_p_name] = updated_incidence_mat.shape[0] - 1
    
    if new_t_name == None:
        # get a name for the new tau
        tau_num_nested_l = [re.findall("(?!tau)\d+", tau_name) for tau_name in trans_dict.keys()]
        tau_num_l = [int(tau_num) for sublist in tau_num_nested_l for tau_num in sublist]

        if len(tau_num_l) == 0:
            new_tau_name = 'tau_0'
        else:
            new_tau_name = 'tau_{}'.format(max(tau_num_l) + 1)

        updated_trans_dict[new_tau_name] = updated_incidence_mat.shape[1] - 1            
    else:
        updated_trans_dict[new_t_name] = updated_incidence_mat.shape[1] - 1

#     # make the new t skipable
#     if new_t_skippable or new_t_in_loop:
        
#         if new_t_skippable and new_t_in_loop: # t is in a skippable loop
#             # reverse t's connection
#             updated_incidence_mat[:,-1] = -updated_incidence_mat[:,-1]
#             updated_incidence_mat = np.concatenate((updated_incidence_mat, 
#                                     -updated_incidence_mat[:,-1].reshape(-1, 1)), axis=1)
#         if new_t_skippable and (not new_t_in_loop):
#             updated_incidence_mat = np.concatenate((updated_incidence_mat, 
#                                                     updated_incidence_mat[:,-1].reshape(-1, 1)), axis=1)
#         elif (not new_t_skippable) and new_t_in_loop:
#             updated_incidence_mat = np.concatenate((updated_incidence_mat, 
#                                                     -updated_incidence_mat[:,-1].reshape(-1, 1)), axis=1)

#         # get a name for the new tau
#         tau_num_nested_l = [re.findall("(?!tau)\d+", tau_name) for tau_name in trans_dict.keys()]
#         tau_num_l = [int(tau_num) for sublist in tau_num_nested_l for tau_num in sublist]

#         if len(tau_num_l) == 0:
#             new_tau_name = 'tau_0'
#         else:
#             new_tau_name = 'tau_{}'.format(max(tau_num_l) + 1)

#         updated_trans_dict[new_tau_name] = updated_incidence_mat.shape[1] - 1
        
    if new_t_skippable or new_t_in_loop:

        if new_t_in_loop:                
            # build a new with skippable == False, loop == True first
            # if skippable == True, loop == True, swap the new_t_name, and new tau later
            new_S = set([updated_places_dict[new_p_name]])
            new_R = set(np.where(updated_incidence_mat[updated_places_dict[new_p_name],:] == -1)[0])
            updated_incidence_mat, updated_places_dict, updated_trans_dict = apply_new_rule(
                                                                                incidence_mat=updated_incidence_mat, 
                                                                                new_t_name=None,
                                                                                R=new_R,
                                                                                S=new_S,
                                                                                places_dict=updated_places_dict, 
                                                                                trans_dict=updated_trans_dict, 
                                                                                new_t_skippable=False, 
                                                                                new_t_in_loop=False)
            updated_incidence_mat = np.concatenate((updated_incidence_mat, 
                                                    -updated_incidence_mat[:,updated_trans_dict[new_t_name]].reshape(-1, 1)), axis=1)

        else: # skippable == True, loop == False
            updated_incidence_mat = np.concatenate((updated_incidence_mat, 
                                                    updated_incidence_mat[:,-1].reshape(-1, 1)), axis=1)


        # get a name for the new tau
        tau_num_nested_l = [re.findall("(?!tau)\d+", tau_name) for tau_name in updated_trans_dict.keys()]
        tau_num_l = [int(tau_num) for sublist in tau_num_nested_l for tau_num in sublist]

        if len(tau_num_l) == 0:
            new_tau_name = 'tau_0'
        else:
            new_tau_name = 'tau_{}'.format(max(tau_num_l) + 1)

        if new_t_skippable and new_t_in_loop: # swap the new_t_name and new tau in the updated_trans_dict
            updated_trans_dict[new_tau_name] = updated_trans_dict[new_t_name]
            updated_trans_dict[new_t_name] = updated_incidence_mat.shape[1] - 1
        else:
            updated_trans_dict[new_tau_name] = updated_incidence_mat.shape[1] - 1

    return updated_incidence_mat, updated_places_dict, updated_trans_dict


def add_silent_transitions(net_dict):

    net = copy.deepcopy(net_dict['petri'])
    
    m = incidence_matrix.construct(net)
    incidence_mat = np.array(m.a_matrix)
    trans_dict = {k.name: int(v) for k, v in m.transitions.items()}
    places_dict = {k.name: int(v) for k, v in m.places.items()}
    
    # for transitions with more than two out_args
    R_l = [t_i for t_i, b in enumerate(np.sum(incidence_mat == 1, axis=0) >= 2) if b]
    S_l = [[p_i for p_i, b in enumerate(b_l) if b] for b_l in [(incidence_mat[:,t_i] == 1) for t_i in R_l]]

    R_S_pairs = [(set([R_l[i]]), set(S_l[i])) for i in range(len(R_l))]
    
    updated_incidence_mat = incidence_mat.copy()
    updated_places_dict = places_dict.copy()
    updated_trans_dict = trans_dict.copy()
    
    for pair in R_S_pairs:
        
            
        R = pair[0]
        S = pair[1]
        
        # only apply abstraction on labelled transitions
        if not ([k for k, v in m.transitions.items() if v == list(R)[0]][0].label == None):
            updated_incidence_mat, updated_places_dict, updated_trans_dict = apply_abstraction_rule(
                                                                                incidence_mat=updated_incidence_mat, 
                                                                                new_t_name=None, 
                                                                                R=R, 
                                                                                S=S, 
                                                                                places_dict=updated_places_dict, 
                                                                                trans_dict=updated_trans_dict)
            
    # for transitions with more than two in_args
    R_l = [t_i for t_i, b in enumerate(np.sum(incidence_mat == -1, axis=0) >= 2) if b]
    S_l = [[p_i for p_i, b in enumerate(b_l) if b] for b_l in [(incidence_mat[:,t_i] == -1) for t_i in R_l]]

    S_R_pairs = [(set(S_l[i]), set([R_l[i]])) for i in range(len(R_l))]
    
    updated_incidence_mat = updated_incidence_mat.copy()
    updated_places_dict =  updated_places_dict.copy()
    updated_trans_dict = updated_trans_dict.copy()
    
    for pair in S_R_pairs:
        
            
        S = pair[0]
        R = pair[1]
        
        # only apply abstraction on labelled transitions
        if not ([k for k, v in m.transitions.items() if v == list(R)[0]][0].label == None):
            updated_incidence_mat, updated_places_dict, updated_trans_dict = apply_new_rule(updated_incidence_mat, 
                                                                                        new_t_name=None, 
                                                                                        R=R, 
                                                                                        S=S, 
                                                                                        places_dict=updated_places_dict, 
                                                                                        trans_dict=updated_trans_dict)
    
    new_petri_dict = create_pn_from_incidence_mat(updated_incidence_mat, updated_places_dict, updated_trans_dict)
    break_sc_petri = remove_tran_by_name(new_petri_dict['petri'], trans_name='short_circuited_transition')
    
    # update incidence_mat, places and trans dict
    # since the incidence_matrix.construct() sort the nodes by name
#     new_petri_m = incidence_matrix.construct(new_petri)
#     new_petri_mat = np.array(new_petri_m.a_matrix)
#     new_petri_places_dict = {k.name: int(v) for k, v in new_petri_m.places.items()}
#     new_petri_trans_dict = {k.name: int(v) for k, v in new_petri_m.transitions.items()}

    net_dict.update(new_petri_dict)
    net_dict['break_sc_petri'] = break_sc_petri
#     net_dict['petri'] = new_petri_dict['petri']
#     net_dict['incidence_mat'] = new_petri_dict['incidence_mat']
#     net_dict['places_dict'] = new_petri_dict['places_dict']
#     net_dict['trans_dict'] = new_petri_dict['trans_dict']

    

    for p in break_sc_petri.places:
        if p.name == 'source':
            p_source = p
            im = Marking({p_source: 1})
        elif p.name == 'sink':
            p_sink = p
            fm = Marking({p_sink: 1})

    net_dict['im'] = im
    net_dict['fm'] = fm
    
    return net_dict

# updated
def get_all_new_nets(net, new_t_name:str, tbc_nodes_inds_dict=None, use_new_rule=False):

    m = incidence_matrix.construct(net)
    mat = np.array(m.a_matrix)
    trans_dict = {k.name: int(v) for k, v in m.transitions.items()}
    places_dict = {k.name: int(v) for k, v in m.places.items()}

    source_ind = places_dict['source']
    sink_ind = places_dict['sink']
    
    # print(trans_dict)

    start_tau_ind = trans_dict['start_tau']
    end_tau_ind = trans_dict['end_tau']
    short_circuit_ind = trans_dict['short_circuited_transition']
    
    new_trans = get_new_nodes_gen_by_LD_rules(mat, source_ind=source_ind, sink_ind=sink_ind, start_tau_ind=start_tau_ind, 
                                              end_tau_ind=end_tau_ind, short_circuit_ind=short_circuit_ind, node_type='t', 
                                              tbc_nodes_inds_dict=tbc_nodes_inds_dict)
    
    new_places = get_new_nodes_gen_by_LD_rules(mat, source_ind=source_ind, sink_ind=sink_ind, start_tau_ind=start_tau_ind, 
                                               end_tau_ind=end_tau_ind, short_circuit_ind=short_circuit_ind, node_type='p', 
                                               tbc_nodes_inds_dict=tbc_nodes_inds_dict)
    # print('get_new_nodes_gen_by_LD_p_rules')
    possible_abs_pairs_gen = get_possible_abs_pairs_gen(mat, source_ind=source_ind, sink_ind=sink_ind, 
                                                        tbc_nodes_inds_dict=tbc_nodes_inds_dict)
    # print('possible_abs_pairs_gen')
    if use_new_rule:
        new_rule_pairs = get_all_new_rule_pairs(mat, source_ind=source_ind, sink_ind=sink_ind, 
                                                tbc_nodes_inds_dict=tbc_nodes_inds_dict)
    else:
        new_rule_pairs = []
    
    new_nets_list = []
    
    net_ind = 0
    
    # print('applying_LD_t_rule')
    for new_t_col in new_trans:
        
        updated_incidence_mat, updated_trans_dict = apply_LD_t_rule(incidence_mat=mat, new_t_name=new_t_name, 
                                                                    new_t_col=new_t_col, trans_dict=trans_dict)
        new_net_dict = {
            'incidence_mat': updated_incidence_mat, 
            'trans_dict': updated_trans_dict, 
            'places_dict': places_dict,
            'rule': 'linear dependency t'
        }
        
        new_nets_list.append(new_net_dict)
        
        net_ind += 1
    
    # print('applying_LD_p_rule')
    for new_p_row in new_places:
        for new_t_in_loop, new_t_skippable in product([False,True], repeat=2):
            updated_incidence_mat, updated_places_dict, updated_trans_dict = apply_LD_p_then_abs_rules(incidence_mat=mat,
                                                                                                       new_t_name=new_t_name,
                                                                                                       new_p_row=new_p_row,
                                                                                                       places_dict=places_dict,
                                                                                                       trans_dict=trans_dict, 
                                                                                                       new_t_skippable=new_t_skippable, 
                                                                                                       new_t_in_loop=new_t_in_loop)
            new_net_dict = {
                'incidence_mat': updated_incidence_mat, 
                'trans_dict': updated_trans_dict, 
                'places_dict': updated_places_dict,
                'rule': 'linear dependency p + abstraction'
            }
            new_nets_list.append(new_net_dict)
            net_ind += 1
    
    # print('apply_abstraction_rule')
    for ab_pair in possible_abs_pairs_gen:
        R, S = ab_pair[0], ab_pair[1]
        
        for new_t_in_loop, new_t_skippable in product([False,True], repeat=2):
            updated_incidence_mat, updated_places_dict, updated_trans_dict = apply_abstraction_rule(incidence_mat=mat, 
                                                                                                    new_t_name=new_t_name, 
                                                                                                    R=R, 
                                                                                                    S=S, 
                                                                                                    trans_dict=trans_dict, 
                                                                                                    places_dict=places_dict, 
                                                                                                    new_t_skippable=new_t_skippable, 
                                                                                                    new_t_in_loop=new_t_in_loop)
            new_net_dict = {
                'incidence_mat': updated_incidence_mat, 
                'trans_dict': updated_trans_dict, 
                'places_dict': updated_places_dict,
                'rule': 'abstraction + (linear dependent t)'
            }
            new_nets_list.append(new_net_dict)
            net_ind += 1
    
    # print('apply_new_rule')
    for new_pair in new_rule_pairs:
        S, R = new_pair[0], new_pair[1]
        
        for new_t_in_loop, new_t_skippable in product([False,True], repeat=2):
            updated_incidence_mat, updated_places_dict, updated_trans_dict = apply_new_rule(incidence_mat=mat, 
                                                                                            new_t_name=new_t_name, 
                                                                                            R=R, 
                                                                                            S=S, 
                                                                                            trans_dict=trans_dict, 
                                                                                            places_dict=places_dict, 
                                                                                            new_t_skippable=new_t_skippable, 
                                                                                            new_t_in_loop=new_t_in_loop)
            new_net_dict = {
                'incidence_mat': updated_incidence_mat, 
                'trans_dict': updated_trans_dict, 
                'places_dict': updated_places_dict,
                'rule': 'new rule'
            }
            new_nets_list.append(new_net_dict)
            net_ind += 1
        
    return new_nets_list


def check_if_optional_loop(activity_name:str, activity_name_log):
    
    optional = not all([activity_name in trace for trace in activity_name_log])
    loop = any([sum(activity_name == np.array(trace)) > 1 for trace in activity_name_log])
    
    return optional, loop

def get_caus_rel_dict(new_t_name:str, log, c = 1, w = 0, rel_1_v = 0, noise_threshold=0.2):
    
    unrolled_log = [a['concept:name'] for trace in log for a in trace]
    
    uniq_a, uniq_a_count = np.unique(unrolled_log, return_counts=True)
    uniq_a_count_dict = {uniq_a[i]:c for i, c in enumerate(uniq_a_count)}

    dfg = dfg_inst.apply(log)
    
    def filter_dfg(dfg, log, new_t_name, noise_threshold=0.2):

        end_activities = [[a['concept:name'] for a in trace][-1] for trace in log]
        end_a, end_a_edge_weights = np.unique(end_activities, return_counts=True)
        end_a_freq_dict = {k:0 for k in set([k[0] for k in dfg] + [k[1] for k in dfg])}
        for i, a in enumerate(end_a):
            end_a_freq_dict[a] = end_a_edge_weights[i]

        start_activities = [[a['concept:name'] for a in trace][0] for trace in log]
        start_a, start_a_edge_weights = np.unique(start_activities, return_counts=True)
        start_a_freq_dict = {k:0 for k in set([k[0] for k in dfg] + [k[1] for k in dfg])}
        for i, a in enumerate(start_a):
            start_a_freq_dict[a] = start_a_edge_weights[i]

        out_arcs_threshold = max([v for k, v in dfg.items() 
                                      if k[0] == new_t_name] + [end_a_freq_dict[new_t_name]])*noise_threshold

        in_arcs_threshold = max([v for k, v in dfg.items() 
                                  if k[1] == new_t_name] + [start_a_freq_dict[new_t_name]])*noise_threshold

        # filter the dfg only for new_t
        filtered_dfg = {k:v for k, v in dfg.items() if (new_t_name not in k) 
                                                    or (k[0] == new_t_name and v >= out_arcs_threshold) 
                                                    or (k[1] == new_t_name and v >= in_arcs_threshold)}

        return filtered_dfg

    dfg = filter_dfg(dfg=dfg, log=log, new_t_name=new_t_name, noise_threshold=noise_threshold)
    

    if rel_1_v == 0:
        t_new_star_sum = np.sum([v for k, v in dfg.items() if new_t_name == k[0]])
        star_t_new_sum = np.sum([v for k, v in dfg.items() if new_t_name == k[1]])
    elif rel_1_v == 1:
        t_new_star_sum = uniq_a_count_dict[new_t_name]
        star_t_new_sum = uniq_a_count_dict[new_t_name]
    
    rel_1_dict = {}
    rel_2_dict = {}

    # construct relaltion 1 & 2 dictionary
    for pair, count in dfg.items():

        if new_t_name == pair[0]: # if new_t_name is the first activity of the pair
            t_existing_name = pair[1]
            if rel_1_v == 0:
                # calculate the times some activities are followed by the second activity of the pair 
                star_t_existing_sum = np.sum([v for k, v in dfg.items() if t_existing_name == k[1]])
            elif rel_1_v == 1:
                star_t_existing_sum = uniq_a_count_dict[t_existing_name]

            rel_1_dict[pair] = (2*count)/(t_new_star_sum + star_t_existing_sum)
            # for relation 2 calculation
            n2e_count = count
            e2n_count = np.sum([v for k, v in dfg.items() if ((t_existing_name == k[0]) and (new_t_name == k[1]))])

            if n2e_count - e2n_count > 0:
                rel_2_n2e = (n2e_count - e2n_count)/(n2e_count + e2n_count + c)
            elif pair[0] == pair[1]:
                rel_2_n2e = n2e_count/(n2e_count + c)
            else: 
                rel_2_n2e = 0

            rel_2_dict[pair] = rel_2_n2e
        elif new_t_name == pair[1]: # if new_t_name is the 2nd activity of the pair
            t_existing_name = pair[0]
            if rel_1_v == 0: # sum up all the counts where 
                t_existing_star_sum = np.sum([v for k, v in dfg.items() if t_existing_name == k[0]])
            elif rel_1_v == 1:
                t_existing_star_sum = uniq_a_count_dict[t_existing_name]

            rel_1_dict[pair] = (2*count)/(t_existing_star_sum + star_t_new_sum)

            e2n_count = count
            n2e_count = np.sum([v for k, v in dfg.items() if ((t_existing_name == k[1]) and (new_t_name == k[0]))])

            if e2n_count - n2e_count > 0:
                rel_2_e2n = (e2n_count - n2e_count)/(e2n_count + n2e_count + c)
            elif pair[0] == pair[1]:
                rel_2_e2n = e2n_count/(e2n_count + c)
            else: 
                rel_2_e2n = 0

            rel_2_dict[pair] = rel_2_e2n

    caus_rel_dict = {rel_1_k:w*rel_1_v + (1-w)*rel_2_dict[rel_1_k] for rel_1_k, rel_1_v in rel_1_dict.items()}
    
    return caus_rel_dict


def get_adjacent_activities(new_t_name:str, caus_rel_dict:dict, t_Rs=0.2):
    
    concurrent_pairs = []
    concurrent_activities = []
    pre_activities = []
    post_activities = []
    
    caus_rel_dict = {k:v for k, v in caus_rel_dict.items() if v >= t_Rs}
    # print('filtered_caus_rel_dict:', caus_rel_dict)
    
    for k, v in caus_rel_dict.items():
        
        # concurrent
        if ((k[1], k[0]) in caus_rel_dict.keys()) and set(k) not in concurrent_pairs:
            concurrent_pairs.append(set(k))
            
#         # filter concurrent pairs
#         concurrent_pairs_rel_mix = [[rel_mix for pair, rel_mix in caus_rel_dict.items() if set(pair) == c_p] 
#                                     for c_p in concurrent_pairs]
#         concurrent_pairs = [c_p for i, c_p in enumerate(concurrent_pairs) 
#                             if abs(concurrent_pairs_rel_mix[i][0] - concurrent_pairs_rel_mix[i][1]) < 0.5]
        
        # preceding activities
        if k[1] == new_t_name and v >= t_Rs:
            pre_activities.append(k[0])

        # post activities: following activities
        if k[0] == new_t_name and v >= t_Rs:
            post_activities.append(k[1])
    
#     print(concurrent_pairs, "before filter")
    # print(pre_activities)
    # print(post_activities)
    
    if all([len(p) == 2 for p in concurrent_pairs]): # if there is no self loop, e.g. concurrent pair ('t', 't')
        concurrent_activities = set([list(p.difference(set([new_t_name])))[0] for p in concurrent_pairs])
    else: # if there is a self loop, e.g. concurrent pair ('t', 't')
        concurrent_activities = set([list(p.difference(set([new_t_name])))[0] for p in concurrent_pairs if len(p) == 2])
        concurrent_activities.add(new_t_name)
    pre_activities = set([a for a in pre_activities if a not in concurrent_activities])
    post_activities = set([a for a in post_activities if a not in concurrent_activities])
    
    adjacent_act_dict = {
        'concurrent': concurrent_activities,
        'pre': pre_activities,
        'post': post_activities
    }
    
    return adjacent_act_dict

def get_nodes_on_the_path(net, source_node_names:set, target_node_names:set, 
                          start_tau_name='start_tau', end_tau_name='end_tau'):
    
    def get_adjacent_nodes_in_pm(net, node_name, direction='source'):
        # assume the node_name should be in either net.transitions or net.places
        # check the node type
        if node_name in [t.name for t in net.transitions]:
            node = [t for t in net.transitions if t.name == node_name][0]
        else:
            node = [p for p in net.places if p.name == node_name][0]

        if direction == 'source':
            adjacent_nodes = set([arc.source.name for arc in node.in_arcs])
        elif direction == 'target':
            adjacent_nodes = set([arc.target.name for arc in node.out_arcs])
        elif direction == 'both':
            adjacent_nodes = set([arc.source.name for arc in node.in_arcs]).union(set([arc.target.name 
                                                                                       for arc in node.out_arcs]))
        return adjacent_nodes
    
    G, G_label = create_networkx_directed_graph(net)
    
    if len(source_node_names) == 0:
        source_node_names.add(start_tau_name)
    if len(target_node_names) == 0:
        target_node_names.add(end_tau_name)
    
    # print(G, G_label)
    # print(source_node_names, target_node_names)
    
    trans_on_the_path = []
    places_on_the_path = []
    all_simple_paths = []
    # print(G_label.items())
    for source_target_pair in product(source_node_names, target_node_names):
        source_node_name, target_node_name = source_target_pair
        
        # print(source_node_name, target_node_name)
        # print([node for node, pn_node in G_label.items() if type(pn_node)==pm4py.objects.petri_net.obj.PetriNet.Transition and pn_node.name == source_node_name])
        # print([node for node, pn_node in G_label.items() if type(pn_node)==pm4py.objects.petri_net.obj.PetriNet.Transition and pn_node.name == target_node_name])
        # source_in_G = [node for node, pn_node in G_label.items() if type(pn_node)==pm4py.objects.petri_net.obj.PetriNet.Transition and pn_node.name == source_node_name][0]
        if len([node for node, pn_node in G_label.items() if type(pn_node)==pm4py.objects.petri_net.obj.PetriNet.Transition and pn_node.name == target_node_name]) > 0:
            target_in_G = [node for node, pn_node in G_label.items() if type(pn_node)==pm4py.objects.petri_net.obj.PetriNet.Transition and pn_node.name == target_node_name][0]
        else:
            print('Target: ', target_node_name)
            print([node for node, pn_node in G_label.items() if type(pn_node)==pm4py.objects.petri_net.obj.PetriNet.Transition and pn_node.name == target_node_name])
            print(G_label.items())
            sys.exit(0)
        if len([node for node, pn_node in G_label.items() if type(pn_node)==pm4py.objects.petri_net.obj.PetriNet.Transition and pn_node.name == source_node_name]) > 0:
            source_in_G = [node for node, pn_node in G_label.items() if type(
                pn_node) == pm4py.objects.petri_net.obj.PetriNet.Transition and pn_node.name == source_node_name][0]
        else:
            print('Source: ', source_node_name)
            print([node for node, pn_node in G_label.items() if type(
                pn_node) == pm4py.objects.petri_net.obj.PetriNet.Transition and pn_node.name == source_node_name])
            print(G_label.items())
            sys.exit(0)
        all_simple_paths.append([[G_label[i] for i in path]
                                 for path in nx.all_simple_paths(G, source=source_in_G, target=target_in_G)])
#     print(all_simple_paths)
    
    # unpack all_simple_paths
    all_simple_paths = [path for paths in all_simple_paths for path in paths]
    nodes_names_on_the_path = [[n.name for n in path] for path in all_simple_paths]
    # filter out simple paths that need to go through short_circuit transition
    paths_mask = ['short_circuited_transition' not in path for path in nodes_names_on_the_path]
    all_simple_paths = [all_simple_paths[i] for i, b in enumerate(paths_mask) if b]
#     print(all_simple_paths)

    # in case there are some cycles on the path
    cycles = [[G_label[n] for n in l] for l in nx.algorithms.cycles.simple_cycles(G)]
    
    # filter boolean cycles that contain "short_circuit"
    cycles_b_ind = ['short_circuited_transition' not in c for c in [[n.name for n in c] for c in cycles]]
    cycles = [set(c) for i, c in enumerate(cycles) if cycles_b_ind[i]]
#     print(cycles)

    # if both source and target nodes are not in C, then include the cycle
    cycles_mask = [all([((len(set(P).intersection(C)) != 0) and (len(set([P[0], P[-1]]).intersection(C)) == 0)) 
                        for P in all_simple_paths]) 
                   for C in cycles]
    cycles = [cycles[i] for i, b in enumerate(cycles_mask) if b]
#     print(cycles_mask, cycles)
    
    # unpack all_simple_paths to a set of nodes
    nodes_on_the_path = set([node for path in all_simple_paths for node in path])
    
    trans_on_the_path = set([node.name for node in nodes_on_the_path if type(node) == PetriNet.Transition])
    places_on_the_path = set([node.name for node in nodes_on_the_path if type(node) == PetriNet.Place])
    
#     print(trans_on_the_path, places_on_the_path)
    
    if len(cycles) > 0: # unpack cycles to a set of nodes
        nodes_in_cycles = set([node for cycle in cycles for node in cycle])
        trans_in_cycles = set([node.name for node in nodes_in_cycles if type(node) == PetriNet.Transition])
        places_in_cycles = set([node.name for node in nodes_in_cycles if type(node) == PetriNet.Place])
        # union nodes_on_the_path and nodes_in_cycles
        trans_on_the_path = trans_on_the_path.union(trans_in_cycles)
        places_on_the_path = places_on_the_path.union(places_in_cycles)
    
    # if there are no nodes on the path, most likely the new t is the first activiy to be added in a loop
    if len(trans_on_the_path) + len(places_on_the_path) == 0:
        trans_on_the_path.update(source_node_names.union(target_node_names))
        sourceSource_node_names = [get_adjacent_nodes_in_pm(net, node_name=node_name, direction='both') 
                                   for node_name in source_node_names]
        sourceSource_node_names = set([n for l in sourceSource_node_names for n in l])
        targetTarget_node_names = [get_adjacent_nodes_in_pm(net, node_name=node_name, direction='both') 
                                   for node_name in target_node_names]
        targetTarget_node_names = set([n for l in targetTarget_node_names for n in l])
        
        places_on_the_path.update(sourceSource_node_names.union(targetTarget_node_names))
    
    nodes_on_the_path_dict = {
        'transitions': trans_on_the_path,
        'places': places_on_the_path
    }
    
    return nodes_on_the_path_dict

def get_tbc_nodes_inds_dict(net, new_t_name:str, log, t_Rs=0.6, noise_threshold=0.2):
    
#     if break_sc:
#         net = remove_tran_by_name(net, trans_name='short_circuit')
    
    m = incidence_matrix.construct(net)
    mat = np.array(m.a_matrix)
    trans_dict = {k.name: int(v) for k, v in m.transitions.items()}
    places_dict = {k.name: int(v) for k, v in m.places.items()}

    # try to get the nodes that are most likely to be used for adding new nodes
    caus_rel_dict = get_caus_rel_dict(new_t_name=new_t_name, log=log, noise_threshold=noise_threshold)
    caus_rel_dict_0_noise = get_caus_rel_dict(new_t_name=new_t_name, log=log, noise_threshold=0)
    # print('caus_rel_dict:', caus_rel_dict)
    
    adjacent_act_dict = get_adjacent_activities(new_t_name=new_t_name, caus_rel_dict=caus_rel_dict, t_Rs=t_Rs)
    adjacent_act_dict_0_noise = get_adjacent_activities(new_t_name=new_t_name, caus_rel_dict=caus_rel_dict_0_noise, t_Rs=t_Rs)
    # print('adjacent_act_dict:', adjacent_act_dict)
    
    nodes_on_the_path_dict = get_nodes_on_the_path(net=net, 
                                                   source_node_names=adjacent_act_dict['pre'], 
                                                   target_node_names=adjacent_act_dict['post'])
    # print('nodes_on_the_path_dict:', nodes_on_the_path_dict)
    
    tbc_nodes_inds_dict = {
        'transitions': [trans_dict[t] for t in nodes_on_the_path_dict['transitions']],
        'places': [places_dict[p] for p in nodes_on_the_path_dict['places']],
        'pre': [trans_dict[t] for t in adjacent_act_dict_0_noise['pre']],
        'post': [trans_dict[t] for t in adjacent_act_dict_0_noise['post']]
    }

    # print('caus_rel_dict:', caus_rel_dict)
    # print('caus_rel_dict_0_noise:', caus_rel_dict_0_noise)
    # print('adjacent_act_dict:', adjacent_act_dict)
    # print('nodes_on_the_path_dict:', nodes_on_the_path_dict)
    print('trans_dict:', trans_dict)
    print('places_dict:', places_dict)
    # print('tbc_nodes_inds_dict:', tbc_nodes_inds_dict)
    
    return tbc_nodes_inds_dict


def get_the_best_net(net, new_t_name, log, net_dict=None, theta=0.9, visualize=False, 
                     use_heuristics=True, t_Rs=0.2, use_recall=False, 
                     noise_threshold=0.2,use_new_rule=False, unblock_mode=False):
    
    # activity_name_log = [[a['concept:name'] for a in trace] for trace in log]
    
#     def net_2_net_dict(net, sc_tran_name='short_circuited_transition', visualize=False):
        
#         break_sc_petri = remove_tran_by_name(net, trans_name=sc_tran_name)
#         # update incidence_mat, places and trans dict
#         # since the incidence_matrix.construct() sort the nodes by name
#         new_petri_m = incidence_matrix.construct(net)
#         new_petri_mat = np.array(new_petri_m.a_matrix)
#         new_petri_places_dict = {k.name: int(v) for k, v in new_petri_m.places.items()}
#         new_petri_trans_dict = {k.name: int(v) for k, v in new_petri_m.transitions.items()}
#         net_dict['petri'] = net
#         net_dict['incidence_mat'] = new_petri_mat
#         net_dict['places_dict'] = new_petri_places_dict
#         net_dict['trans_dict'] = new_petri_trans_dict

#         net_dict['break_sc_petri'] = break_sc_petri

#         for p in break_sc_petri.places:
#             if p.name == 'source':
#                 p_source = p
#                 im = Marking({p_source: 1})
#             elif p.name == 'sink':
#                 p_sink = p
#                 fm = Marking({p_sink: 1})

#         net_dict['im'] = im
#         net_dict['fm'] = fm
        
#         if visualize:
#             # visualization
#             gviz = pn_visualizer.apply(break_sc_petri, im, fm)
#             pn_visualizer.view(gviz)
#             print(net_dict['rule'])        

#         try:
#             fitness_dict = replay_fitness_evaluator.apply(log, break_sc_petri, im, fm, 
#                                                             variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)
#         except Exception:
#             fitness_dict = {'percFitTraces': 0,
#                             'averageFitness': 0,
#                             'percentage_of_fitting_traces': 0,
#                             'average_trace_fitness': 0,    
#                             'log_fitness': 0}
#             pass
#         net_dict['fitness_dict'] = fitness_dict
#         recall = fitness_dict['percentage_of_fitting_traces']/100
#         net_dict['recall'] = recall
        
#         return net_dict
                
    
    if use_heuristics:
        # try to get the nodes that are most likely to be used for adding new nodes
        tbc_nodes_inds_dict = get_tbc_nodes_inds_dict(net=net,
                                                      new_t_name=new_t_name, 
                                                      log=log, 
                                                      t_Rs=t_Rs, 
                                                      noise_threshold=noise_threshold)

        n_tbc_nodes = len(tbc_nodes_inds_dict['transitions']) + len(tbc_nodes_inds_dict['places'])
    else:
        tbc_nodes_inds_dict = None
        n_tbc_nodes = 0

    n_nodes_last = len(net.transitions) + len(net.places) - 1

    ratio_tbc_nodes = n_tbc_nodes/n_nodes_last

    if n_tbc_nodes > 50:
        the_chosen_net_dict = {}
        the_chosen_net_dict['n_nodes'] = n_nodes_last
        the_chosen_net_dict['n_nodes_last'] = n_nodes_last
        the_chosen_net_dict['n_tbc_nodes'] = n_tbc_nodes
        the_chosen_net_dict['ratio_tbc_nodes'] = ratio_tbc_nodes
        the_chosen_net_dict['time_for_finding_nets'] = 0
        the_chosen_net_dict['n_nets'] = 0
        fitness_dict = {'log_fitness': 0}
        the_chosen_net_dict['fitness_dict'] = fitness_dict
        the_chosen_net_dict['recall'] = 0
        the_chosen_net_dict['precision'] = 0
        the_chosen_net_dict['F1'] = 0
        the_chosen_net_dict['break_sc_petri'] = None
        the_chosen_net_dict['IM'] = None
        return the_chosen_net_dict
    t_before = time.perf_counter()
    new_nets_list = get_all_new_nets(net=net, new_t_name=new_t_name, tbc_nodes_inds_dict=tbc_nodes_inds_dict,use_new_rule=use_new_rule)
    t_after = time.perf_counter()
    t_diff_nets = t_after - t_before
    # evaluate the new nets
    counter = 0
    for net_dict in new_nets_list:
        new_petri = create_pn_from_incidence_mat(incidence_mat=net_dict['incidence_mat'], 
                                                      places_dict=net_dict['places_dict'], 
                                                      trans_dict=net_dict['trans_dict'], 
                                                      return_net_dict=False)

        break_sc_petri = remove_tran_by_name(new_petri, trans_name='short_circuited_transition')

        # update incidence_mat, places and trans dict
        # since the incidence_matrix.construct() sort the nodes by name
        new_petri_m = incidence_matrix.construct(new_petri)
        new_petri_mat = np.array(new_petri_m.a_matrix)
        new_petri_places_dict = {k.name: int(v) for k, v in new_petri_m.places.items()}
        new_petri_trans_dict = {k.name: int(v) for k, v in new_petri_m.transitions.items()}
        net_dict['petri'] = new_petri
        net_dict['incidence_mat'] = new_petri_mat
        net_dict['places_dict'] = new_petri_places_dict
        net_dict['trans_dict'] = new_petri_trans_dict

        net_dict['break_sc_petri'] = break_sc_petri

        for p in break_sc_petri.places:
            if p.name == 'source':
                p_source = p
                im = Marking({p_source: 1})
            elif p.name == 'sink':
                p_sink = p
                fm = Marking({p_sink: 1})

        net_dict['im'] = im
        net_dict['fm'] = fm
        pm4py.save_vis_petri_net(break_sc_petri, im, fm, 'temp_' + str(counter) + '.png')
        counter += 1
        if visualize:
            # visualization
            gviz = pn_visualizer.apply(break_sc_petri, im, fm)
            pn_visualizer.view(gviz)
            print(net_dict['rule'])        

        try:
            fitness_dict = replay_fitness_evaluator.apply(log, break_sc_petri, im, fm, 
                                                            variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)
        except Exception:
            fitness_dict = {'percFitTraces': 0,
                            'averageFitness': 0,
                            'percentage_of_fitting_traces': 0,
                            'average_trace_fitness': 0,    
                            'log_fitness': 0}
            pass
        net_dict['fitness_dict'] = fitness_dict
        recall = fitness_dict['percentage_of_fitting_traces']/100
        net_dict['recall'] = recall
        
        net_dict = add_silent_transitions(net_dict)
        
        if visualize:
    
            print(net_dict['fitness_dict'])
    
    ################# start selection ##################
    # get the new nets with >= threshold
    if use_recall:
        cand_nets_dict = [new_net_dict for new_net_dict in new_nets_list 
                                        if new_net_dict['recall'] >= theta]
    else:
        cand_nets_dict = [new_net_dict for new_net_dict in new_nets_list 
                                        if new_net_dict['fitness_dict']['log_fitness'] >= theta]
    
    # in case there are multiple nets that have >= threshold
    if len(cand_nets_dict) > 1:
        if visualize:
            print('######################################################')
            print("evaluating precision")

        for cand_net_dict in cand_nets_dict:
            precision = precision_evaluator.apply(log, 
                                                  cand_net_dict['break_sc_petri'], 
                                                  cand_net_dict['im'], 
                                                  cand_net_dict['fm'], 
                                                  variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
            cand_net_dict['precision'] = precision
            
            log_fitness = cand_net_dict['fitness_dict']['log_fitness']
            
            if use_recall:
                cand_net_dict['F1'] = 2*((precision * cand_net_dict['recall'])/(precision + cand_net_dict['recall']))
            else: 
                cand_net_dict['F1'] = 2*((precision * log_fitness)/(precision + log_fitness))
            
            if visualize:

                print("precision", cand_net_dict['precision'])
                gviz = pn_visualizer.apply(cand_net_dict['break_sc_petri'], 
                                           cand_net_dict['im'], 
                                           cand_net_dict['fm'])
                pn_visualizer.view(gviz)
        
        max_F1 = max([cand_net_dict['F1'] for cand_net_dict in cand_nets_dict])
        cand_nets_dict = [cand_net_dict for cand_net_dict in cand_nets_dict 
                              if cand_net_dict['F1'] == max_F1]
        
        # in case there are more than one net having max F1
        if len(cand_nets_dict) > 1:
            for final_cand_net_dict in cand_nets_dict:
                n_elements = len(final_cand_net_dict['break_sc_petri'].transitions) +\
                            len(final_cand_net_dict['break_sc_petri'].places) +\
                            len(final_cand_net_dict['break_sc_petri'].arcs)
                final_cand_net_dict['n_elements'] = n_elements

                if visualize:
                    print('######################################################')
                    print("nets with same max precision")
                    print("precision", final_cand_net_dict['precision'])
                    gviz = pn_visualizer.apply(final_cand_net_dict['break_sc_petri'], 
                                               final_cand_net_dict['im'], 
                                               final_cand_net_dict['fm'])
                    pn_visualizer.view(gviz)

            # in case there are more than one net have min element
            min_elements_n = min([final_cand_net_dict['n_elements'] for final_cand_net_dict in cand_nets_dict])
            # if len([final_cand_net_dict 
            #         for final_cand_net_dict in cand_nets_dict 
            #             if final_cand_net_dict['n_elements'] == min_elements_n]) > 1:
            #     # print("more than one net with the same min elements")
                
            if visualize:
                for final_cand_net_dict in cand_nets_dict:
                    if final_cand_net_dict['n_elements'] == min_elements_n:
                        print('######################################################')
                        gviz = pn_visualizer.apply(final_cand_net_dict['break_sc_petri'], 
                                                   final_cand_net_dict['im'], 
                                                   final_cand_net_dict['fm'])
                        pn_visualizer.view(gviz)

            the_chosen_net_ind = np.argmin([final_cand_net_dict['n_elements'] 
                                            for final_cand_net_dict in cand_nets_dict])
            the_chosen_net_dict = cand_nets_dict[the_chosen_net_ind]
        else:
            the_chosen_net_dict = cand_nets_dict[0]
            
    elif len(cand_nets_dict) == 0: # there is no candidate net has >= threshold
        if unblock_mode and net_dict:
            return net_dict
        else:
            # then fallback
            
            return get_the_best_net(net, new_t_name, log, theta=theta, noise_threshold=noise_threshold,
                                    visualize=visualize, use_heuristics=False, use_recall=use_recall)
    else: # only one can_net
        the_chosen_net_dict = cand_nets_dict[0]
        # need to evaluate precision for the one
        precision = precision_evaluator.apply(log, 
                                              the_chosen_net_dict['break_sc_petri'], 
                                              the_chosen_net_dict['im'], 
                                              the_chosen_net_dict['fm'], 
                                              variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
        the_chosen_net_dict['precision'] = precision

        log_fitness = the_chosen_net_dict['fitness_dict']['log_fitness']

        if use_recall:
            recall = the_chosen_net_dict['recall']
            the_chosen_net_dict['F1'] = 2*((precision * recall)/(precision + recall))
        else: 
            the_chosen_net_dict['F1'] = 2*((precision * log_fitness)/(precision + log_fitness))
        
#     the_chosen_net_dict = add_silent_transitions(the_chosen_net_dict)

    the_chosen_net_dict['n_nodes'] = len(the_chosen_net_dict['trans_dict']) + len(the_chosen_net_dict['places_dict']) - 1
    the_chosen_net_dict['n_nodes_last'] = n_nodes_last
    the_chosen_net_dict['n_tbc_nodes'] = n_tbc_nodes
    the_chosen_net_dict['ratio_tbc_nodes'] = ratio_tbc_nodes
    the_chosen_net_dict['time_for_finding_nets'] = t_diff_nets
    the_chosen_net_dict['n_nets'] = len(new_nets_list)

    return the_chosen_net_dict

def remove_redundant_taus(net_dict, final=True, use_new_rule=False):
       
    def check_redundant_taus(net_dict, final=True, use_new_rule=False):
              
        if use_new_rule:
            direction = 1
        else:
            direction = -1
        
        incidence_mat = copy.deepcopy(net_dict['incidence_mat'])
        # get tau transitions except for sc, start and end taus
        tau_indx = np.array([v for k, v in net_dict['trans_dict'].items() if bool(re.search("tau_\d+", k))])
        
        if len(tau_indx) == 0:
            return False
        # get the transitions having only one input (output) place
        tau_indx_mask = np.sum((incidence_mat[:, tau_indx] == direction), axis=0) == 1
        
        if not final:
            # when reducing the nets not in the final, rule out silent transitions with >= 2 out_arcs (in_arcs)
            tau_indx_mask_2 = np.sum((incidence_mat[:, tau_indx] == -direction), axis=0) == 1
            tau_indx_mask = np.array([b and tau_indx_mask_2[i] for i, b in enumerate(tau_indx_mask)])
            
        tau_indx = tau_indx[tau_indx_mask]
        if len(tau_indx) == 0:
            return False
        # get the corresponding places having only one output transition
        P_indx = np.array([p_i for i in tau_indx for p_i, v in enumerate(incidence_mat[:,i]) if v == direction])
        P_indx_mask = np.sum(incidence_mat[P_indx] == direction, axis=1) == 1
        if not any(P_indx_mask):
            return False
        P_indx = P_indx[P_indx_mask]
        tau_indx = tau_indx[P_indx_mask]
        
        # final filter
        # filter out the pairs if any of the "tau_postpostset" is connected back to p (a loop of length 1)
        tau_postset_indx = [list(np.where(incidence_mat[:,tau_ind] == -direction)[0]) for tau_ind in tau_indx]
        tau_postpostset_indx = [([list(np.where(p_row == direction)[0]) for p_row in incidence_mat[tau_postset_ind,:]]) 
                                for tau_postset_ind in tau_postset_indx]
        tau_postpostset_indx = [[indx_3 for indx_2 in indx for indx_3 in indx_2] for indx in tau_postpostset_indx]
#         print(tau_postpostset_indx)
        final_mask = [np.all(incidence_mat[:,t_indx][P_indx[i_P_indx]] != -direction) 
                      for i_P_indx, t_indx in enumerate(tau_postpostset_indx)]
#         print(final_mask)
        P_indx = P_indx[final_mask]
        tau_indx = tau_indx[final_mask]

        pt_pairs_list = [[p, tau_indx[i]] for i, p in enumerate(P_indx)]
        
        is_fc_mask = [reduce_by_abstraction(net_dict['incidence_mat'], 
                                            p_ind=pt_pair[0], 
                                            t_ind=pt_pair[1], 
                                            use_new_rule=use_new_rule, 
                                            check_fc_mode=True) 
                      for pt_pair in pt_pairs_list]
        
        pt_pairs_list = [pt_pair for i, pt_pair in enumerate(pt_pairs_list) if is_fc_mask[i]]
        
        return pt_pairs_list
    
    def reduce_by_abstraction(incidence_mat, p_ind, t_ind, use_new_rule=False, check_fc_mode=False):
        
        updated_incidence_mat = incidence_mat.copy()
        
        p_row = updated_incidence_mat[p_ind,:]
        t_col = updated_incidence_mat[:,t_ind]
        
        if use_new_rule:
            
            p_postset_indx = [ind for ind, v in enumerate(p_row) if v == -1]
            t_preset_indx = [ind for ind, v in enumerate(t_col) if v == -1]

            # break the connections from p_postset to p
            for p_postset_i in p_postset_indx:
                updated_incidence_mat[p_ind,p_postset_i] = 0
            # break the connections from t to t_postset
            for t_preset_i in t_preset_indx:
                updated_incidence_mat[t_preset_i,t_ind] = 0           
            # connnect p_preset to t_postset
            for indx_pair in product(t_preset_indx, p_postset_indx):
                updated_incidence_mat[indx_pair[0], indx_pair[1]] = -1 
        else:
            
            p_preset_indx = [ind for ind, v in enumerate(p_row) if v == 1]
            t_postset_indx = [ind for ind, v in enumerate(t_col) if v == 1]

            # break the connections from p_preset to p
            for p_preset_i in p_preset_indx:
                updated_incidence_mat[p_ind,p_preset_i] = 0
            # break the connections from t to t_postset
            for t_postset_i in t_postset_indx:
                updated_incidence_mat[t_postset_i,t_ind] = 0           
            # connnect p_preset to t_postset
            for indx_pair in product(t_postset_indx, p_preset_indx):
                updated_incidence_mat[indx_pair[0], indx_pair[1]] = 1
                
        # remove p and t from the incidence_mat_mask
        n_places, n_trans = updated_incidence_mat.shape
        incidence_mat_mask = np.full((n_places, n_trans), True)
        incidence_mat_mask[p_ind, :] = False
        incidence_mat_mask[:,t_ind] = False
        updated_incidence_mat = updated_incidence_mat[incidence_mat_mask].reshape(n_places-1, n_trans-1)
        
        if check_fc_mode:
            return is_fc_net_from_mat(updated_incidence_mat)
        else:    
            return updated_incidence_mat

    
    updated_net_dict = copy.deepcopy(net_dict)
    
#     check_redundant_taus(updated_net_dict, use_new_rule=use_new_rule)
    
    pt_pairs_list = check_redundant_taus(updated_net_dict, final=final, use_new_rule=use_new_rule)
    
        
    while pt_pairs_list:
        
        p_ind = pt_pairs_list[0][0]
        tau_ind = pt_pairs_list[0][1]
        updated_net_dict['incidence_mat'] = reduce_by_abstraction(incidence_mat=updated_net_dict['incidence_mat'], 
                                                                  p_ind=p_ind, 
                                                                  t_ind=tau_ind, 
                                                                  use_new_rule=use_new_rule)

        # update places_dict and trans_dict
        updated_trans = [k for k, v in sorted(updated_net_dict['trans_dict'].items(), key=lambda item: item[1]) 
                            if v != tau_ind]
        updated_places = [k for k, v in sorted(updated_net_dict['places_dict'].items(), key=lambda item: item[1]) 
                              if v != p_ind]

        updated_net_dict['trans_dict'] = {tran:ind for ind, tran in enumerate(updated_trans)}
        updated_net_dict['places_dict'] = {place:ind for ind, place in enumerate(updated_places)}
        
        
        pt_pairs_list = check_redundant_taus(updated_net_dict, final=final, use_new_rule=use_new_rule)
        
    items_tb_updated = create_pn_from_incidence_mat(incidence_mat=updated_net_dict['incidence_mat'], 
                                                          places_dict=updated_net_dict['places_dict'], 
                                                          trans_dict=updated_net_dict['trans_dict'], 
                                                          return_net_dict=True)
    
    # since the incidence_matrix.construct() sort the nodes by name, update the dict
    updated_net_dict.update(items_tb_updated)
        
    return updated_net_dict

def get_DFG_from_petri(net, im):
    
    def get_neighbor_trans(transition, explore_out=True):

        # if 'None' in transition.name:
        #     raise TypeError('can only process non-silent transition')
        if explore_out:

            out_trans = []
            explored_tau_trans = []
            for out_t in transition.to_state.outgoing: 
                if 'None' not in out_t.name:
                    out_trans.append(out_t)
                else:
                    if out_t not in explored_tau_trans:
                        explored_tau_trans.append(out_t)
                        out_trans += get_neighbor_trans(out_t, explore_out=explore_out)

            return out_trans

        else:
            in_trans = []
            explored_tau_trans = []
            for in_t in transition.from_state.incoming: 
                if 'None' not in in_t.name:
                    in_trans.append(in_t)
                else:
                    if in_t not in explored_tau_trans:
                        explored_tau_trans.append(in_t)
                        in_trans += get_neighbor_trans(in_t, explore_out=explore_out)

            return in_trans
        
    rg = reachability_graph.construct_reachability_graph(net, im)
    DFG_edges = []
    # get transition names uniquely
    tran_names = [re.search("(?<=').*(?='\))|None(?=\))", t.name).group() for t in rg.transitions if 'None' not in t.name]
    tran_names = list(set(tran_names))

    # 
    for t_name in tran_names:
        # print([t for t in rg.transitions if re.search(f"'{t_name}'", t.name)])
        # print(t_name)
        trans_wth_same_name = [t for t in rg.transitions if re.search(f"'{t_name}'", t.name)]

        if len(trans_wth_same_name) == 1:
            out_trans = get_neighbor_trans(trans_wth_same_name[0], explore_out=True)
            in_trans = get_neighbor_trans(trans_wth_same_name[0], explore_out=False)
        else:
            out_trans = [get_neighbor_trans(tran, explore_out=True) for tran in trans_wth_same_name]
            out_trans = [tran for sublist in out_trans for tran in sublist]

            in_trans = [get_neighbor_trans(tran, explore_out=False) for tran in trans_wth_same_name]
            in_trans = [tran for sublist in in_trans for tran in sublist]


        in_tran_names = [re.search("(?<=').*(?='\))|None(?=\))", tran.name).group() for tran in in_trans]
        out_tran_names = [re.search("(?<=').*(?='\))|None(?=\))", tran.name).group() for tran in out_trans]

        t_out_edges = list(product(set([t_name]), set(out_tran_names)))
        t_in_edges = list(product(set(in_tran_names), set([t_name])))

        DFG_edges.append(t_out_edges)
        DFG_edges.append(t_in_edges)

    DFG_edges = [edge for sublist in DFG_edges for edge in sublist]
    DFG_edges = set(DFG_edges)
    # print(set(DFG_edges))
    
    return DFG_edges

def is_p_net(net):
    return all(
        [((len(t.in_arcs) == 1) and (len(t.out_arcs) == 1)) for t in net.transitions]
    )

def is_t_net(net):
    return all([((len(p.in_arcs) == 1) and (len(p.out_arcs) == 1)) for p in net.places])

def get_similarity_score(net, im, log, noise_threshold=0.2):
    
    def filter_dfg(log, noise_threshold=0.2):

        dfg_dict, start_activities, end_activities = pm4py.discover_dfg(log)
        # filter out going arcs
        threshold_dict = {a:max([v for k, v in dfg_dict.items() if k[0] == a])*noise_threshold 
                          for a in set([k[0] for k in dfg_dict])}

        for k, v in end_activities.items():
            if k not in threshold_dict.keys():
                threshold_dict[k] = v*noise_threshold

        filtered_dfg_dict = {k:v for k, v in dfg_dict.items() if v >= threshold_dict[k[0]]}

        return filtered_dfg_dict, start_activities, end_activities
    
    DFG_from_petri = get_DFG_from_petri(net, im)
    # DFG_from_log, start_activities, end_activities = pm4py.discover_dfg(log)
    
    DFG_from_log, start_activities, end_activities = filter_dfg(log=log, noise_threshold=noise_threshold)
    
    nodes_in_DFG_from_log = []
    nodes_in_DFG_from_petri = []
    for edge in DFG_from_log:
        nodes_in_DFG_from_log += list(edge)

    for edge in DFG_from_petri:
        nodes_in_DFG_from_petri += list(edge)
    
    nodes_in_DFG_from_log = set([n for n in set(nodes_in_DFG_from_log) if len([e for e in DFG_from_log if e[1]==n]) >= 1]).union(set(start_activities.keys()))
    nodes_in_DFG_from_petri = set(nodes_in_DFG_from_petri)

    nodes_intersect = nodes_in_DFG_from_log.intersection(nodes_in_DFG_from_petri)
    # print(nodes_intersect)
    
    # get preset_threshold_dict and postset_threshold_dict
    preset_threshold_dict = {a:max([v for k, v in DFG_from_log.items() if k[1] == a])*noise_threshold for a in set([k[1] for k in DFG_from_log])}
    for k, v in start_activities.items():
        preset_threshold_dict[k] = v*noise_threshold

    postset_threshold_dict = {a:max([v for k, v in DFG_from_log.items() if k[0] == a])*noise_threshold for a in set([k[0] for k in DFG_from_log])}
    for k, v in end_activities.items():
        postset_threshold_dict[k] = v*noise_threshold

    pre_post_dict_log = {}
    pre_post_dict_petri = {}
    for node in nodes_intersect:

        pre_post_dict_log[node] = dict(preset=set([k[0] for k,v in DFG_from_log.items() if (k[1] == node) and (v > preset_threshold_dict[node])]), 
                                       postset=set([k[1] for k,v in DFG_from_log.items() if (k[0] == node) and (v > preset_threshold_dict[node])]))
        
        pre_post_dict_petri[node] = dict(preset=set([i[0] for i in DFG_from_petri if i[1] == node]), 
                                         postset=set([i[1] for i in DFG_from_petri if i[0] == node]))

    print(pre_post_dict_log)
    print(pre_post_dict_petri)
    for key, value in pre_post_dict_petri.items():
        if len(value['preset']) == 0:
            pre_post_dict_petri[key]['preset'] = set(['start_tau'])
        if len(value['postset']) == 0:
            pre_post_dict_petri[key]['postset'] = set(['end_tau'])

    for key, value in pre_post_dict_log.items():
        if len(value['preset']) == 0:
            pre_post_dict_log[key]['preset'] = set(['start_tau'])
        if len(value['postset']) == 0:
            pre_post_dict_log[key]['postset'] = set(['end_tau'])

    similarity_score_dict = {}
    # max_possible_arcs = max(
    #     [(len(value['preset']) + len(value['postset'])) for key, value in pre_post_dict_log.items()])
    max_possible_arcs = max(
        [len(pre_post_dict_petri[node]['preset'].union(pre_post_dict_log[node]['preset'])) +
         len(pre_post_dict_petri[node]['postset'].union(pre_post_dict_log[node]['postset']))
         for node in pre_post_dict_log.keys()]
    )
    max_possible_arcs += 1
    print('Max arc:', max_possible_arcs)
    for node in nodes_intersect:

        len_preset_intersect = len(pre_post_dict_petri[node]['preset'].intersection(pre_post_dict_log[node]['preset']))
        len_preset_union = len(pre_post_dict_petri[node]['preset'].union(pre_post_dict_log[node]['preset']))
        len_postset_intersect = len(pre_post_dict_petri[node]['postset'].intersection(pre_post_dict_log[node]['postset']))
        len_postset_union = len(pre_post_dict_petri[node]['postset'].union(pre_post_dict_log[node]['postset']))
        print(node, len_preset_intersect, len_preset_union, len_postset_intersect, len_postset_union)
        # TODO modify function ?
        similarity_score_dict[node] = (1/2)*(len_preset_intersect/len_preset_union) + (1/2)*(len_postset_intersect/len_postset_union)
        # similarity_score_dict[node] = compute_efficiency_penalty_score({
        #     "len_preset_intersect": len_preset_intersect,
        #     "len_preset_union": len_preset_union,
        #     "len_postset_intersect": len_postset_intersect,
        #     "len_postset_union": len_postset_union
        # }, max_possible_arcs)
    # print(similarity_score_dict)
    return similarity_score_dict

def compute_efficiency_penalty_score(node, max_possible_arcs):
    efficiency = (node["len_preset_intersect"] + node["len_postset_intersect"]) / (node["len_preset_union"] + node["len_postset_union"])
    penalty_raw = 1 - (node["len_preset_union"] + node["len_postset_union"]) / max_possible_arcs
    penalty = max(0, min(1, penalty_raw))  # Ensure penalty is in the range [0, 1]
    return efficiency * penalty


def reduce_net(node, net, initial_marking):
    def get_t_induced_tnet(tbrm_t, tbrm_p=None):
        if tbrm_p == None:
            tbrm_p = set()

        new_tbrm_t = tbrm_t.copy()
        for t in tbrm_t:
            for a_tp in t.out_arcs:
                if len(a_tp.target.in_arcs) == 1 and len(a_tp.target.out_arcs) == 1:
                    tbrm_p.add(a_tp.target)
                    for a_pt in a_tp.target.out_arcs:
                        new_tbrm_t.add(a_pt.target)

        if len(new_tbrm_t) != len(tbrm_t):
            return get_t_induced_tnet(new_tbrm_t, tbrm_p)
        else:
            return tbrm_t, tbrm_p

    def get_p_induced_pnet(tbrm_p, tbrm_t=None):
        if tbrm_t == None:
            tbrm_t = set()

        new_tbrm_p = tbrm_p.copy()
        for p in tbrm_p:
            for a_tp in p.in_arcs:
                if len(a_tp.source.in_arcs) == 1 and len(a_tp.source.out_arcs) == 1:
                    tbrm_t.add(a_tp.source)
                    for a_pt in a_tp.source.in_arcs:
                        new_tbrm_p.add(a_pt.source)

        if len(new_tbrm_p) != len(tbrm_p):
            return get_p_induced_pnet(new_tbrm_p, tbrm_t)
        else:
            return tbrm_t, tbrm_p

    new_initial_marking = initial_marking.copy()

    # in case input is already a p(t)_net and it is requested a p(t)_net
    if ((type(node) == PetriNet.Place) and is_p_net(net)) or (
        (type(node) == PetriNet.Transition) and is_t_net(net)
    ):
        proper = False
        return net, initial_marking, proper
    elif type(node) == PetriNet.Place:
        tbrm_t, tbrm_p = get_p_induced_pnet(set([node]))
    elif type(node) == PetriNet.Transition:
        tbrm_t, tbrm_p = get_t_induced_tnet(set([node]))

        # calcualte new marking in case of t-reduction
        # Proposition 4 & Definition 12
        enabled_t_hat = semantics.enabled_transitions(
            net, new_initial_marking
        ) & tbrm_t - set([node])
        while enabled_t_hat != set():
            for t in enabled_t_hat:
                new_initial_marking = semantics.execute(t, net, new_initial_marking)
                enabled_t_hat = semantics.enabled_transitions(
                    net, new_initial_marking
                ) & tbrm_t - set([node])
    else:
        raise TypeError("Only a Place/Transition is allowed")

    # get tbrm_a
    tbrm_a = set()
    for a in net.arcs:
        if (set([a.source, a.target]) & (tbrm_t | tbrm_p)) != set():
            tbrm_a.add(a)

    new_net = PetriNet("new_petri_net")

    new_net.transitions.update(net.transitions.copy().difference(tbrm_t))
    new_net.places.update(net.places.copy().difference(tbrm_p))
    new_net.arcs.update(net.arcs.copy().difference(tbrm_a))

    pnml_exporter.pnml.export_net(
        petrinet=new_net, marking=new_initial_marking, output_filename="temp_net.pnml"
    )
    new_net, new_initial_marking, final_marking = pnml_importer.apply("./temp_net.pnml")

    # check if proper (complement is strongly connected)
    networkx, inv_dictionary = create_networkx_directed_graph(new_net)
    
    # update to only allow removing max one labeled transition
    proper = is_strongly_connected(networkx) and (len([t for t in tbrm_t if t.label != None]) <= 1)

    return new_net, new_initial_marking, proper

def toggle_sct(net, initial_marking, final_marking=None):

    # verify if free choice net and strongly connected
    networkx, _ = create_networkx_directed_graph(net)
    is_sc = is_strongly_connected(networkx)
    # is_fc_net = is_free_choice_net(net)

    if not is_sc:
        # print("Input Petri net is not strongly connected free-choice net.")
        new_net = short_circuit_petri_net(net)[0]

        # change short circuit transition name
        # [t for t in net.transitions if t.name == "short_circuited_transition"][0].name = 'sct'

        # fix new initial marking because of short circuit
        initial_marking_p_names = [p.name for p in initial_marking.elements()]
        new_initial_markings_p = [p for p in new_net.places if p.name in initial_marking_p_names]
        new_initial_marking = Marking()
        for p in new_initial_markings_p:
            new_initial_marking[p] = 1

        initial_marking = new_initial_marking.copy()
        
    else:

        if 'short_circuited_transition' not in [t.name for t in net.transitions]:
            return net, initial_marking, final_marking
        else:
            # print([t for t in net.transitions])

            new_net = copy.deepcopy(net)
            sct = [t for t in new_net.transitions if t.name == 'short_circuited_transition'][0]
            sink = [arc for arc in sct.in_arcs][0].source
            source = [arc for arc in sct.out_arcs][0].target
            
            final_marking = Marking()
            initial_marking = Marking()
            final_marking[sink] = 1
            initial_marking[source] = 1

            tbrm_a = set()

            for a in new_net.arcs:
                if (set([a.source, a.target]) & set([sct])) != set():
                    tbrm_a.add(a)

            for a in tbrm_a:
                new_net.arcs.remove(a)

            new_net.transitions.remove(sct)

            pnml_exporter.pnml.export_net(petrinet=new_net, marking=initial_marking, final_marking=final_marking, output_filename="temp_net.pnml")
            
            new_net, initial_marking, final_marking = pnml_importer.apply("./temp_net.pnml")
    
    return new_net, initial_marking, final_marking

def is_free_choice_net(net):

    t_comb = combinations(net.transitions, 2)

    b_list = []
    for i in list(t_comb):
        in_nodes_t1 = set([a.source for a in i[0].in_arcs])
        in_nodes_t2 = set([a.source for a in i[1].in_arcs])

        is_in_nodes_same = in_nodes_t1 == in_nodes_t2
        is_in_nodes_disjoint = in_nodes_t1.intersection(in_nodes_t2) == set()

        b_list.append(is_in_nodes_same or is_in_nodes_disjoint)

    return all(b_list)

def reduce_by_abstraction(tran_label, net, initial_marking, dual=False):
    
    net, initial_marking, final_marking = toggle_sct(net=net, initial_marking=initial_marking)
    # pm4py.view_petri_net(net, initial_marking, final_marking)
    transition = [t for t in net.transitions if t.label == tran_label][0]
    if not dual:
        transition_targets = [arc.target for arc in transition.out_arcs]
        in_place = [arc.source for arc in transition.in_arcs][0]
        in_place_sources = [arc.source for arc in in_place.in_arcs]
        net = remove_transition(net=net, trans=transition)
        net = remove_place(net=net, place=in_place)
        
        for fr_to in product(in_place_sources, transition_targets): 
            add_arc_from_to(fr=fr_to[0], to=fr_to[1], net=net)
        new_net, new_im, new_fm = toggle_sct(net=net, initial_marking=initial_marking)
        is_fc = is_free_choice_net(new_net)
        
        return new_net, new_im, new_fm, is_fc
    
    else:
        transition_sources = [arc.source for arc in transition.in_arcs]
        out_place = [arc.target for arc in transition.out_arcs][0]
        out_place_targets = [arc.target for arc in out_place.out_arcs]
        net = remove_transition(net=net, trans=transition)
        net = remove_place(net=net, place=out_place)
        
        for fr_to in product(transition_sources, out_place_targets): 
            add_arc_from_to(fr=fr_to[0], to=fr_to[1], net=net)
        new_net, new_im, new_fm = toggle_sct(net=net, initial_marking=initial_marking)
        is_fc = is_free_choice_net(new_net)

        return new_net, new_im, new_fm, is_fc


def remove_a_transition(tran_label, net, initial_marking):
    sc_net, initial_marking, _ = toggle_sct(net=net, initial_marking=initial_marking)
    # print([t for t in sc_net.transitions])
    # print(tran_label)
    transition = [t for t in sc_net.transitions if t.label == tran_label][0]
    in_places = [arc.source for arc in transition.in_arcs]
    out_places = [arc.target for arc in transition.out_arcs]
    sct = [t for t in sc_net.transitions if t.name == 'short_circuited_transition'][0]
    sink = [arc for arc in sct.in_arcs][0].source
    source = [arc for arc in sct.out_arcs][0].target
    # source = [p for p in net.places if len(p.in_arcs) == 0][0]
    out_place = [arc.target for arc in transition.out_arcs][0]

    # check if there is a silent transition has the same pre and postset as the tbr_transition
    # equivalent to Fusion of Parallel Transitions (FPT) for removing silent transition
    n_same_prePost_silent_tran = len(
        [t for t in sc_net.transitions if ([a.source for a in t.in_arcs] == [a.source for a in transition.in_arcs] and
                                           [a.target for a in t.out_arcs] == [a.target for a in transition.out_arcs] and
                                           t.label == None)])
    if n_same_prePost_silent_tran >= 1:
        new_net, new_im, t_reduction_proper = reduce_net(node=transition, net=sc_net, initial_marking=initial_marking)
    else:
        t_reduction_proper = False
    if t_reduction_proper:
        # remove sct
        new_net, new_im, new_fm = toggle_sct(net=new_net, initial_marking=new_im)
        new_net = apply_fpt_rule(new_net)
        new_net = apply_simple_reduction(new_net)
        return new_net, new_im, new_fm

    # # try p reduction
    new_net, new_im, p_reduction_proper = reduce_net(node=out_place, net=sc_net, initial_marking=initial_marking)
    if p_reduction_proper:
        new_net, new_im, new_fm = toggle_sct(net=new_net, initial_marking=new_im)
        new_net = apply_fpt_rule(new_net)
        new_net = apply_simple_reduction(new_net)
        return new_net, new_im, new_fm

    # otherwise, make it silent and remove it using reduction rules
    else:
        # get the transition from net using tran_label
        new_net, _, _ = toggle_sct(net=sc_net, initial_marking=initial_marking)
        transition = [t for t in new_net.transitions if t.label == tran_label][0]
        transition.label = None

        # try normal abstraction in fc book
        # equivalent to Fusion of Series Places (FSP) for removing silent transition, but ensure free-choiceness
        if (len(transition.in_arcs) == 1) and (len([arc.source for arc in transition.in_arcs][0].out_arcs) == 1) and \
                (source not in in_places) and (sink not in in_places) and (len(transition.out_arcs) == 1):
            new_net, new_im, new_fm, is_fc = reduce_by_abstraction(tran_label=tran_label, net=net,
                                                                   initial_marking=initial_marking)
            if is_fc:
                # pm4py.view_petri_net(new_net, new_im, new_fm)
                # new_net, new_im, new_fm = remove_implicit_places.apply_reduction(new_net, new_im, new_fm)
                new_net = apply_fpt_rule(new_net)
                new_net = apply_simple_reduction(new_net)
                return new_net, new_im, new_fm
            else:
                pass

        # try dual abstraction
        # equivalent to Fusion of Series Transitions (FST) for removing silent transition, but ensure free-choiceness
        elif (len(transition.out_arcs) == 1) and (len([arc.target for arc in transition.out_arcs][0].in_arcs) == 1) and \
                (source not in out_places) and (sink not in out_places):
            new_net, new_im, new_fm, is_fc = reduce_by_abstraction(tran_label=tran_label, net=net,
                                                                   initial_marking=initial_marking, dual=True)
            if is_fc:
                # new_net, new_im, new_fm = remove_implicit_places.apply_reduction(new_net, new_im, new_fm)
                new_net = apply_fpt_rule(new_net)
                new_net = apply_simple_reduction(new_net)
                return new_net, new_im, new_fm
            else:
                pass

        final_marking = Marking()
        initial_marking = Marking()
        sink = [p for p in new_net.places if len(p.out_arcs) == 0][0]
        source = [p for p in new_net.places if len(p.in_arcs) == 0][0]
        final_marking[sink] = 1
        initial_marking[source] = 1

        new_net = apply_fpt_rule(new_net)
        new_net = apply_simple_reduction(new_net)
        return new_net, initial_marking, final_marking


def adjust_net_for_SynthesisMiner(net):
    
    new_net = copy.deepcopy(net)
    # get the source and sink place and rename them
    source = [p for p in new_net.places if len(p.in_arcs) == 0][0]
    sink = [p for p in new_net.places if len(p.out_arcs) == 0][0]
    source.name = 'source'
    sink.name = 'sink'
    
    # get start/end transitions
    start_trans = [arc.target for arc in source.out_arcs]
    end_trans = [arc.source for arc in sink.in_arcs]
    
    # if there is only one start/end transition and it is silent transition, then ...
    if len(start_trans) == 1 and start_trans[0].label == None:
        start_trans[0].name = 'start_tau'
    if len(end_trans) == 1 and end_trans[0].label == None:
        end_trans[0].name = 'end_tau'
        
    # rename transitions and places
    tau_transitions = [t for t in new_net.transitions if not t.label and t.name not in ['start_tau', 'end_tau']]
    for i, tau_tran in enumerate(tau_transitions):
        tau_tran.name = 'tau_{}'.format(i)
    for t in new_net.transitions:
        if t.label:
            t.name = t.label
        
    other_places = [p for p in new_net.places if p.name not in ['source', 'sink']]
    for i, p in enumerate(other_places):
        p.name = 'p{}'.format(i)
    
    exist_start_tau = len([t for t in new_net.transitions if t.name == 'start_tau']) == 1
    exist_end_tau = len([t for t in new_net.transitions if t.name == 'end_tau']) == 1
    
    if not exist_start_tau:
        # create a new source and rename the old source
        source.name = 'p{}'.format(len(new_net.places))
        new_source = PetriNet.Place('source'.format(len(new_net.places)))
        new_net.places.add(new_source)
        
        # add start_tau and end_tau and add the arcs from/to the new/old sources
        start_tau = PetriNet.Transition('start_tau', None)
        new_net.transitions.add(start_tau)
        add_arc_from_to(new_source, start_tau, new_net)
        add_arc_from_to(start_tau, source, new_net)

    
    if not exist_end_tau:
        # create a new sink and rename the old sink
        sink.name = 'p{}'.format(len(new_net.places))
        new_sink = PetriNet.Place('sink'.format(len(new_net.places)))
        new_net.places.add(new_sink)
        
        end_tau = PetriNet.Transition('end_tau', None)
        new_net.transitions.add(end_tau)
        add_arc_from_to(sink, end_tau, new_net)
        add_arc_from_to(end_tau, new_sink, new_net)
    
    return new_net
    
    new_net = copy.deepcopy(net)
    # get the source and sink place and rename them
    source = [p for p in new_net.places if len(p.in_arcs) == 0][0]
    sink = [p for p in new_net.places if len(p.out_arcs) == 0][0]
    source.name = 'source'
    sink.name = 'sink'
    
    # get start/end transitions
    start_trans = [arc.target for arc in source.out_arcs]
    end_trans = [arc.source for arc in sink.in_arcs]
    
    # if there is only one start/end transition and it is silent transition, then ...
    if len(start_trans) == 1 and start_trans[0].label == None:
        start_trans[0].name = 'start_tau'
    if len(end_trans) == 1 and end_trans[0].label == None:
        end_trans[0].name = 'end_tau'
        
    # rename transitions and places
    tau_transitions = [t for t in new_net.transitions if not t.label and t.name not in ['start_tau', 'end_tau']]
    for i, tau_tran in enumerate(tau_transitions):
        tau_tran.name = 'tau_{}'.format(i)
    for t in new_net.transitions:
        if t.label:
            t.name = t.label
        
    other_places = [p for p in new_net.places if p.name not in ['source', 'sink']]
    for i, p in enumerate(other_places):
        p.name = 'p{}'.format(i)
    
    exist_start_tau = len([t for t in new_net.transitions if t.name == 'start_tau']) == 1
    exist_end_tau = len([t for t in new_net.transitions if t.name == 'end_tau']) == 1
    
    if not exist_start_tau:
        # remove arcs from the source to start transitions and from end transitions to the sink
        for arc in source.out_arcs:
            new_net.arcs.remove(arc)
        # add start_tau and end_tau
        start_tau = PetriNet.Transition('start_tau', None)
        new_net.transitions.add(start_tau)
        add_arc_from_to(source, start_tau, new_net)
        for start_tran in start_trans:
            p = PetriNet.Place('p{}'.format(len(new_net.places)))
            new_net.places.add(p)
            add_arc_from_to(start_tau, p, new_net)
            add_arc_from_to(p, start_tran, new_net)
    
    if not exist_end_tau:
        for arc in sink.in_arcs:
            new_net.arcs.remove(arc)
        end_tau = PetriNet.Transition('end_tau', None)
        new_net.transitions.add(end_tau)
        add_arc_from_to(end_tau, sink, new_net)       
        for end_tran in end_trans:
            p = PetriNet.Place('p{}'.format(len(new_net.places)))
            new_net.places.add(p)
            add_arc_from_to(end_tran, p, new_net)
            add_arc_from_to(p, end_tau, new_net)
    
    return new_net

def export_results(df, filename, threshold, all_one=False):
    threshold = str(threshold).replace('.', '_')
    if all_one is False:
        filename = 'results_' + filename.split('.')[0] + threshold + '_th.csv'
        resultfile = os.path.join(os.path.dirname(__file__), '..', 'results', 'og', filename)
    else:
        filename = 'results_' + filename.split('.')[0] + threshold + '_th_cut_act.csv'
        resultfile = os.path.join(os.path.dirname(__file__), '..','results', 'og', 'all_one', filename)
    print(df)
    df.to_csv(resultfile)

class TimeoutException(Exception):   # Custom exception class
    pass

def timeout_handler(signum, frame):   # Custom signal handler
    raise TimeoutException



def unblock_IM_model(net, im, fm, log, t_Rs=0.5, use_recall=False, tolerance=0.05, view_final_net=False, noise_threshold=0.2,
                     return_DGs=True, metric_to_track='f1', return_net_dict=False, labels_tbi=None, filename=None, threshold=0, all_one=False):
    
    quality_metrics_dict = dict(precision=[],fitness=[],recall=[],f1=[])
    fitness_dict = replay_fitness_evaluator.apply(log, net, im, fm, variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)
    quality_metrics_dict['fitness'].append(fitness_dict['log_fitness'])
    recall = fitness_dict['percentage_of_fitting_traces']/100
    quality_metrics_dict['recall'].append(recall)
    if not use_recall:
        theta = fitness_dict['log_fitness'] - tolerance
    else:
        theta = recall - tolerance
    
    precision = precision_evaluator.apply(log, net, im, fm,variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
    quality_metrics_dict['precision'].append(precision)
    f1 = 2*((precision * fitness_dict['log_fitness'])/(precision + fitness_dict['log_fitness']))
    quality_metrics_dict['f1'].append(f1)
    
    initial_net_dict = dict(break_sc_petri=net, im=im, fm=fm, fitness_dict=fitness_dict, recall=recall, precision=precision, F1=f1)
    
    # copy new net and create updated new im
    new_net = copy.deepcopy(net)
    im = Marking()
    source = [p for p in new_net.places if len(p.in_arcs) == 0][0]
    im[source] = 1
    
    visable_tran_labels = [t.label for t in net.transitions if t.label]
    
    if 'End' in [t.label for t in net.transitions]:
        tran_tracking_l = set(['End'])
    else:
        tran_tracking_l = set()
    
    if labels_tbi:
        tran_tracking_l = tran_tracking_l.union(labels_tbi)
        
    intermediate_net_dicts = [initial_net_dict]

    results = {}
    i = 1
    while True:
        # print(quality_metrics_dict)
        t_start = time.perf_counter()
        t_start_1 = time.perf_counter()
        # signal.alarm(200)
        if all_one:
            # go through transitions that are not label none and not in tran_tracking_l. use that as tran label
            tran_label = [t.label for t in net.transitions if t.label and t.label not in tran_tracking_l]
            if not tran_label:
                break
            else:
                tran_label = tran_label[0]
                print(tran_label)
        else:
            # step 1
            similarity_score_dict = get_similarity_score(net=new_net, im=im, log=log, noise_threshold=noise_threshold)


            for label in tran_tracking_l:
                if label in similarity_score_dict.keys():
                    del similarity_score_dict[label]
            # print(similarity_score_dict)

            # stopping criterias
            if np.all(np.array([s for s in similarity_score_dict.values()]) == 1.0) or\
            (len(tran_tracking_l) == len(visable_tran_labels)) or\
            (len(quality_metrics_dict[metric_to_track]) >= 3 and np.all(np.diff(quality_metrics_dict[metric_to_track][-3:]) < 0)):
                break

            tran_label = min(similarity_score_dict, key=similarity_score_dict.get)
        tran_tracking_l = tran_tracking_l.union([tran_label])
        # print(tran_label)
        t_end_1 = time.perf_counter()
        t_diff_1 = t_end_1 - t_start_1
        # step 2
        t_start_2 = time.perf_counter()
        if all_one:
            net_copy = copy.deepcopy(net)
            new_net, new_im, new_fm = remove_a_transition(tran_label=tran_label, net=net_copy, initial_marking=im)
        else:
            new_net, new_im, new_fm = remove_a_transition(tran_label=tran_label, net=new_net, initial_marking=im)
        # signal.alarm(0)
        t_end_2 = time.perf_counter()
        t_diff_2 = t_end_2 - t_start_2
        # pm4py.view_petri_net(new_net, new_im, new_fm)

        # step 3
        t_start_3 = time.perf_counter()
        new_net = adjust_net_for_SynthesisMiner(new_net)
        # pm4py.view_petri_net(new_net, new_im, new_fm)
        new_net, new_im, new_fm = toggle_sct(net=new_net, initial_marking=new_im)
        # pm4py.view_petri_net(new_net, new_im, new_fm)
        net_dict = get_the_best_net(net=new_net, new_t_name=tran_label, log=log, use_recall=False, theta=theta, t_Rs=t_Rs,noise_threshold=noise_threshold)
        if net_dict['break_sc_petri'] is None:
            updated_net_dict = net_dict
        else:
            updated_net_dict = remove_redundant_taus(net_dict=net_dict, use_new_rule=False)
            updated_net_dict = remove_redundant_taus(net_dict=net_dict, use_new_rule=True)
        t_end_3 = time.perf_counter()
        t_diff_3 = t_end_3 - t_start_3
        # print(updated_net_dict)
        quality_metrics_dict['fitness'].append(updated_net_dict['fitness_dict']['log_fitness'])
        quality_metrics_dict['recall'].append(updated_net_dict['recall'])
        quality_metrics_dict['precision'].append(updated_net_dict['precision'])
        quality_metrics_dict['f1'].append(updated_net_dict['F1'])
        if all_one is False:
            new_net = updated_net_dict['break_sc_petri']
            im = updated_net_dict['im']
        t_end = time.perf_counter()
        t_diff = t_end - t_start
        results[i] = [t_diff, updated_net_dict['precision'], updated_net_dict['fitness_dict']['log_fitness'], updated_net_dict['F1'], tran_label, t_diff_1, t_diff_2, t_diff_3, updated_net_dict['n_tbc_nodes'], updated_net_dict['n_nodes_last'], updated_net_dict['time_for_finding_nets'], updated_net_dict['n_nets']]
        if filename is not None:
            if all_one is False:
                iter_net_name = os.path.join(os.path.dirname(__file__), '..', 'results', 'nets', 'og',
                                         filename.split('.')[0] + '_' + threshold + '_' + str(i) + '_iter_th.svg')
                pm4py.save_vis_petri_net(updated_net_dict['break_sc_petri'], updated_net_dict['im'], updated_net_dict['fm'], iter_net_name)
        results_df = pd.DataFrame.from_dict(results, orient='index',
                                            columns=['complete_time', 'precision', 'fitness', 'f1', 'cut_act', 't_diff_1', 't_diff_2',
                                                     't_diff_3', 'n_tbc_nodes', 'n_nodes_last', 'time_for_finding_nets', 'n_nets'])
        export_results(results_df, filename, threshold, all_one)
        i += 1
        print('Iteration: ', i)
        print('F1: ', updated_net_dict['F1'])
        # pm4py.view_petri_net(updated_net_dict['break_sc_petri'])
        intermediate_net_dicts.append(updated_net_dict)
        np.save(f'./temp.npy', intermediate_net_dicts)
    results_df = pd.DataFrame.from_dict(results, orient='index', columns=['complete_time','precision', 'fitness', 'f1', 'cut_act', 't_diff_1', 't_diff_2', 't_diff_3', 'n_tbc_nodes', 'n_nodes_last', 'time_for_finding_nets', 'n_nets'])
    final_net_dict_ind = np.argmax([d['F1'] for d in intermediate_net_dicts])
    final_net_dict = intermediate_net_dicts[final_net_dict_ind]
    if all_one is False:
        final_net_dict = remove_redundant_taus(net_dict=final_net_dict, final=True)
        final_net_dict = remove_redundant_taus(net_dict=final_net_dict, final=True, use_new_rule=True)
        final_net_dict['break_sc_petri'] = apply_simple_reduction(final_net_dict['break_sc_petri'])
    if view_final_net:
        pm4py.view_petri_net(final_net_dict['break_sc_petri'], final_net_dict['im'], final_net_dict['fm'])
    
    if return_net_dict:
        return final_net_dict, quality_metrics_dict, intermediate_net_dicts, results_df
    else:
        return final_net_dict['break_sc_petri'], final_net_dict['im'], final_net_dict['fm'], results_df
