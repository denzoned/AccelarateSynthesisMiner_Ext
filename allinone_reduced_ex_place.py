import copy
import sys
import time
from enum import Enum
from itertools import product
import numpy as np
import pandas as pd
import pm4py
import os

from pm4py import convert_to_petri_net, reduce_petri_net_invisibles
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness_evaluator
from pm4py import PetriNet, Marking
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.utils import check_soundness, incidence_matrix
from pm4py.objects.petri_net.utils.check_soundness import check_wfnet
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to
from pm4py.objects.process_tree.obj import ProcessTree
from pm4py.util import constants
from pm4py.objects.petri_net.utils.reduction import apply_fpt_rule, apply_simple_reduction

from utils.place_analysis import place_fitness_old
from utils.synthesis_miner import create_pn_from_incidence_mat, remove_tran_by_name, \
    get_nodes_on_the_path, toggle_sct, get_tbc_nodes_inds_dict, adjust_net_for_SynthesisMiner, get_new_nodes_gen_by_LD_rules, get_possible_abs_pairs_gen, get_all_new_rule_pairs, \
    apply_LD_t_rule, apply_LD_p_then_abs_rules, apply_abstraction_rule, apply_new_rule, remove_a_transition
from utils.murata_reduction_mod import apply_reduction_mod
from pm4py.objects.process_tree.obj import Operator as pt_operator


class Parameters(Enum):
    ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY
    TOKEN_REPLAY_VARIANT = "token_replay_variant"
    CLEANING_TOKEN_FLOOD = "cleaning_token_flood"
    SHOW_PROGRESS_BAR = "show_progress_bar"
    MULTIPROCESSING = "multiprocessing"
    CORES = "cores"


def create_model_pt():
    # create e -> (b, c)
    e = ProcessTree(label="e")
    bc = ProcessTree(operator=pt_operator.SEQUENCE)
    bc.children.append(ProcessTree(label="b"))
    bc.children.append(ProcessTree(label="c"))
    e_bc = ProcessTree(operator=pt_operator.PARALLEL)
    e_bc.children.append(e)
    e_bc.children.append(bc)

    # create (e -> (b, c)) + (d, f)
    df = ProcessTree(operator=pt_operator.PARALLEL)
    df.children.append(ProcessTree(label="d"))
    df.children.append(ProcessTree(label="f"))
    e_bc_df = ProcessTree(operator=pt_operator.SEQUENCE)
    e_bc_df.children.append(e_bc)
    e_bc_df.children.append(df)

    # create a -> ((e -> (b, c)) + (d, f), g) -> h
    root = ProcessTree(operator=pt_operator.SEQUENCE)
    root.children.append(ProcessTree(label="a"))
    root.children.append(ProcessTree(operator=pt_operator.PARALLEL, children=[e_bc_df, ProcessTree(label="g")]))
    root.children.append(ProcessTree(label="h"))

    # print(root)  # prints: a -> ((e -> (b, c)) + (d, f), g) -> h

    return root


def apply_reduction(node, net: PetriNet, im, fm):
    # determine what kind of block the node is part of
    transition = [t for t in net.transitions if t.label == node][0]
    in_places = [arc.source for arc in transition.in_arcs]
    out_places = [arc.target for arc in transition.out_arcs]

    # in_places is 1, out_places is 1, the one in_place has only one in_arc, the one out_place has only one out_arc
    if len(in_places) == 1 and len(out_places) == 1 and len(in_places[0].in_arcs) == 1 and len(
            out_places[0].out_arcs) == 1 and len(in_places[0].out_arcs) == 1 and len(out_places[0].in_arcs) == 1:
        in_trans = [arc.source for arc in in_places[0].in_arcs][0]
        out_trans = [arc.target for arc in out_places[0].out_arcs][0]

        # difference between concurrent and seq.:
        # in_trans has multiple out_arcs, and out_trans has multiple in_arcs, if it does, it is concurrent
        if len(in_trans.out_arcs) > 1 and len(out_trans.in_arcs) > 1:
            print('Removal in concurrent structure')
            net, im, fm = remove_place_arcs(net, im, fm, node)
        elif len(in_trans.out_arcs) == 1 and len(out_trans.in_arcs) == 1:
            print('Removal in sequence structure')
            net, im, fm = remove_place_arcs(net, im, fm, node)
            new_place = PetriNet.Place('p' + str(len(net.places)))
            net.places.add(new_place)

            # Add arc from incoming transition to new place
            add_arc_from_to(in_trans, new_place, net)

            # Add arc from new place to outgoing transition
            add_arc_from_to(new_place, out_trans, net)
        else:
            print('Removal in non-block structure (only one of in_trans/out_trans has one out_arc/in_arc)')
            remove_place_arcs(net, im, fm, node)
            new_place = PetriNet.Place('p' + str(len(net.places)))
            net.places.add(new_place)

            # Add arc from incoming transition to new place
            add_arc_from_to(in_trans, new_place, net)

            # Add arc from new place to outgoing transition
            add_arc_from_to(new_place, out_trans, net)

    # for exclusive choice, make silent and and apply FPT rule, in case now two silent transitions are between the same places
    elif len(in_places) == 1 and len(out_places) == 1 and len(in_places[0].out_arcs) > 1 and len(
            out_places[0].in_arcs) > 1:
        # in_place / out_place has multiple out_arcs / in_arcs, meaning it is part of a exclusive choice
        print('Removal in exclusive choice structure, making it silent')
        transition.label = None
        net = apply_fpt_rule(net)
        # net = apply_simple_reduction(net)
    else:
        print('Removal in loop or non-block structure')
        net, im, fm = remove_a_transition(node, net, im)
    im, fm = find_im_fm(net)
    return net, im, fm


def sync_transition_names_with_labels(net):
    if net is None:
        return net
    for transition in net.transitions:
        if transition.label is not None:
            transition.name = transition.label
    return net


def remove_place_arcs(netin, im=None, fm=None, node=None):
    """
    Remove transition node and its arcs and places, based on conditions
    :param netin:
    :param im:
    :param fm:
    :param node:
    :return: net, im, fm
    """
    netin = sync_transition_names_with_labels(netin)
    # net = copy.deepcopy(netin)
    net = netin
    for transition in net.transitions:
        if transition.name == node:
            # print(transition)
            for in_arc1 in list(transition.in_arcs):
                in_place = in_arc1.source
                in_arc2 = next(iter(in_place.in_arcs))
                if len(in_place.out_arcs) == 1:
                    # only remove incoming place and incoming arc of it only if only one out_arc
                    in_tran = in_arc2.source
                    in_tran.out_arcs.remove(in_arc2)
                    net.arcs.remove(in_arc2)
                    net.places.remove(in_place)
                else:
                    in_place.out_arcs.remove(in_arc1)
                net.arcs.remove(in_arc1)

            for out_arc1 in list(transition.out_arcs):
                out_place = out_arc1.target
                out_arc2 = next(iter(out_place.out_arcs))
                if len(out_place.in_arcs) == 1:
                    out_arc2.target.in_arcs.remove(out_arc2)
                    net.arcs.remove(out_arc2)
                    net.places.remove(out_place)
                else:
                    out_place.in_arcs.remove(out_arc1)
                net.arcs.remove(out_arc1)

            net.transitions.remove(transition)
            return net, im, fm
    return net, im, fm


def find_im_fm(net):
    m = incidence_matrix.construct(net)
    incidence_mat = np.array(m.a_matrix)
    source_ind = np.where(np.all(incidence_mat <= 0, axis=1))[0][0]
    sink_ind = np.where(np.all(incidence_mat >= 0, axis=1))[0][0]
    places_dict = {k.name: int(v) for k, v in m.places.items()}
    source_name = [k for k, v in places_dict.items() if v == source_ind][0]
    sink_name = [k for k, v in places_dict.items() if v == sink_ind][0]
    for p in net.places:
        if p.name == source_name:
            p_source = p
            im = Marking({p_source: 1})
        elif p.name == sink_name:
            p_sink = p
            fm = Marking({p_sink: 1})
    return im, fm


def find_prev_foll_skip_loop(log, t_add, petri_net: PetriNet):
    """
    Find the predecessors and successors of a transition in a log
    :param log:
    :param t_add:
    :param petri_net:
    :return:
    """
    variants = pm4py.get_variants(log)

    prev = set()
    foll = set()
    skip = False  # Initialize as False and set to True if t_add is not found in any variant
    loop = False

    for key, variant in enumerate(variants):
        indexes = [i for i, x in enumerate(variant) if x == t_add]
        if not indexes:
            skip = True  # t_add is not found in the variant

        for index in indexes:
            if index > 0:
                prev.add(variant[index - 1])
            if index < len(variant) - 1:
                foll.add(variant[index + 1])
                # Check if t_add happens in sequence (so a self-loop)
                if variant[index + 1] == t_add:
                    loop = True

    # Double check for whole scope for elements in prev or foll
    prev_to_prev_foll = {elem for elem in prev if any(
        t_add in variant and elem in variant and variant.index(t_add) < variant.index(elem) for variant in variants)}
    foll_to_prev_foll = {elem for elem in foll if any(
        t_add in variant and elem in variant and variant.index(t_add) > variant.index(elem) for variant in variants)}

    # Remove elements that are before and after
    prev -= prev_to_prev_foll
    foll -= foll_to_prev_foll
    prev.discard(t_add)
    foll.discard(t_add)
    return prev, foll, skip, loop


def remove_implicit_place(net, im, fm):
    for place in list(net.places):
        if len(place.in_arcs) == 1 and len(place.out_arcs) == 1:
            in_trans = next(iter(place.in_arcs)).source
            out_trans = next(iter(place.out_arcs)).target
            if len(in_trans.out_arcs) > 1 and len(out_trans.in_arcs) > 1:
                paths = get_nodes_on_the_path(net, {in_trans.name}, {out_trans.name})
                if len(paths['places']) > 1:
                    # remove for in_trans the arc to place, for out_trans the arc from place
                    in_arc = next(iter(place.in_arcs))
                    out_arc = next(iter(place.out_arcs))
                    in_trans.out_arcs.remove(in_arc)
                    out_trans.in_arcs.remove(out_arc)

                    # remove the two arcs from the net and remove the place from the net
                    net.arcs.remove(in_arc)
                    net.arcs.remove(out_arc)
                    net.places.remove(place)
    return net, im, fm


def reduce_net(net):
    net = apply_simple_reduction(net)
    im, fm = find_im_fm(net)
    pm4py.save_vis_petri_net(net, im, fm, file_path='results/nets/temp/red_1.png')
    net = reduce_petri_net_invisibles(net)
    im, fm = find_im_fm(net)
    pm4py.save_vis_petri_net(net, im, fm, file_path='results/nets/temp/red_2.png')

    im, fm = find_im_fm(net)
    net, im, fm = apply_reduction_mod(net, im, fm)
    pm4py.save_vis_petri_net(net, im, fm, file_path='results/nets/temp/red_3.png')

    remove_implicit_place(net, im, fm)

    pm4py.save_vis_petri_net(net, im, fm, file_path='results/nets/temp/red_4.png')
    return net, im, fm


def convert_start_end_tau_to_silent(net: PetriNet, im: Marking, fm: Marking):
    # Find start_tau and end_tau transitions
    start_tau_transition = next((t for t in net.transitions if t.label == 'start_tau'), None)
    end_tau_transition = next((t for t in net.transitions if t.label == 'end_tau'), None)

    # Replace them with silent transitions
    if start_tau_transition:
        start_tau_transition.label = None
        start_tau_transition.name = 'start_tau'
    if end_tau_transition:
        end_tau_transition.label = None
        end_tau_transition.name = 'end_tau'

    return net, im, fm


def get_tbc_nodes_new_dict(net, new_t_name: str, pre_post_dict: dict, log: EventLog):
    m = incidence_matrix.construct(net)
    mat = np.array(m.a_matrix)
    trans_dict = {k.name: int(v) for k, v in m.transitions.items()}
    places_dict = {k.name: int(v) for k, v in m.places.items()}

    # Create reversed dictionaries for easy lookup
    rev_trans_dict = {v: k for k, v in trans_dict.items()}
    # rev_places_dict = {v: k for k, v in places_dict.items()}

    # Initialize place-to-transitions mapping
    place_to_transitions = {}

    if not pre_post_dict['pre']:
        pre_post_dict['pre'] = {'start_tau'}
    if not pre_post_dict['post']:
        pre_post_dict['post'] = {'end_tau'}

    pre_tran = analyze_pre_tran(log, pre_post_dict['pre'])
    post_tran = analyze_foll_tran(log, pre_post_dict['post'])
    pre_tran = [trans_dict[t] for t in pre_tran]
    post_tran = [trans_dict[t] for t in post_tran]

    pre_places = [i for i in range(mat.shape[0]) if any(mat[i, t] > 0 for t in pre_tran)]
    post_places = [i for i in range(mat.shape[0]) if any(mat[i, t] < 0 for t in post_tran)]

    nodes_on_the_path_dict = get_nodes_on_the_path(net, pre_post_dict['pre'], pre_post_dict['post'])
    places_on_path = [places_dict[p] for p in nodes_on_the_path_dict['places']]

    pre_places = [p for p in pre_places if p in places_on_path]
    post_places = [p for p in post_places if p in places_on_path]

    input_trans_of_pre_places = set()
    output_trans_of_post_places = set()
    input_trans_of_post_places = set()

    for place_idx in pre_places:
        input_trans = [t for t in range(mat.shape[1]) if mat[place_idx, t] > 0]
        input_trans_of_pre_places.update(input_trans)

    for place_idx in post_places:
        input_trans = [t for t in range(mat.shape[1]) if mat[place_idx, t] > 0]
        input_trans_of_post_places.update(input_trans)

    all_transitions = list(set(pre_tran + post_tran))
    all_transitions = list(all_transitions + list(input_trans_of_pre_places))
    all_transitions = list(all_transitions + list(input_trans_of_post_places))


    if set(pre_post_dict['pre']) == {'start_tau'} or not pre_tran:
        pre_places = places_on_path

    if set(pre_post_dict['post']) == {'end_tau'} or not post_tran:
        post_places = places_on_path

    all_places = list(set(pre_places + post_places))

    for place_idx in places_on_path:
        # place_name = rev_places_dict[place_idx]
        outgoing_trans = [rev_trans_dict[t] for t in range(mat.shape[1]) if mat[place_idx, t] < 0]
        incoming_trans = [rev_trans_dict[t] for t in range(mat.shape[1]) if mat[place_idx, t] > 0]

        place_type = []
        if place_idx in pre_places:
            place_type.append('output_of_pre')
        elif place_idx in post_places:
            place_type.append('input_of_post')
        else:
            place_type.append('normal')

        place_to_transitions[place_idx] = {
            'incoming': incoming_trans,
            'outgoing': outgoing_trans,
            'place_type': place_type
        }

    tbc_nodes_new_dict = {
        'transitions': list(set(all_transitions)),
        'places': all_places,
        'pre': pre_tran,
        'post': post_tran,
        'place_to_transitions': place_to_transitions,
        'places_on_path': places_on_path,
        'init_place_count': len(places_on_path)
    }

    return tbc_nodes_new_dict


def get_consistent_relations(log, trans_set):
    variants = pm4py.get_variants(log)
    consistent_relations = {t: {'preceding': set(), 'following': set()} for t in trans_set}

    for t in trans_set:
        for key, variant in enumerate(variants):
            for index, t in enumerate(variant):
                if t in trans_set:
                    consistent_relations[t]['preceding'].update(set(variant[:index]) & trans_set)
                    consistent_relations[t]['following'].update(set(variant[index + 1:]) & trans_set)

    for t in trans_set:
        preceding = consistent_relations[t]['preceding']
        following = consistent_relations[t]['following']

        # Find elements in preceding that also appear after t in any variant
        preceding_but_also_following = {
            elem for elem in preceding if any(
                t in variant and elem in variant and variant.index(t) < variant.index(elem) for key,variant in enumerate(variants))
        }

        # Find elements in following that also appear before t in any variant
        following_but_also_preceding = {
            elem for elem in following if any(
                t in variant and elem in variant and variant.index(t) > variant.index(elem) for key,variant in enumerate(variants))
        }

        consistent_relations[t]['preceding'] -= preceding_but_also_following
        consistent_relations[t]['following'] -= following_but_also_preceding
        consistent_relations[t]['preceding'].discard(t)
        consistent_relations[t]['following'].discard(t)

    return consistent_relations


def analyze_pre_tran(log, pre_tran):
    consistent_relations = get_consistent_relations(log, pre_tran)
    removable_transitions = set()

    for t in pre_tran:
        # If a transition in pre_tran is always followed by another transition in pre_tran, mark it as removable
        if consistent_relations[t]['following'] & pre_tran:
            removable_transitions.add(t)

    # Remove the identified transitions from pre_tran
    return pre_tran - removable_transitions


def analyze_foll_tran(log, foll_tran):
    consistent_relations = get_consistent_relations(log, foll_tran)
    removable_transitions = set()

    for t in foll_tran:
        # If a transition in pre_tran is always followed by another transition in pre_tran, mark it as removable
        if consistent_relations[t]['preceding'] & foll_tran:
            removable_transitions.add(t)

    # Remove the identified transitions from pre_tran
    return foll_tran - removable_transitions

# used
def extend_tbc_nodes_inds_dict(tbc_nodes_dict: dict, net, new_t_name: str, pre_post_dict: dict):
    # extend the dict with place_to_transitions
    m = incidence_matrix.construct(net)
    places_dict = {k.name: int(v) for k, v in m.places.items()}
    rev_places_dict = {v: k for k, v in places_dict.items()}
    place_to_transitions = {}

    for place_idx in tbc_nodes_dict['places']:
        place_name = rev_places_dict[place_idx]
        place_obj = next(p for p in net.places if p.name == place_name)

        incoming_trans = [arc.source.name for arc in place_obj.in_arcs]
        outgoing_trans = [arc.target.name for arc in place_obj.out_arcs]

        place_to_transitions[place_idx] = {
            'incoming': incoming_trans,
            'outgoing': outgoing_trans,
            'place_type': 'normal'
        }

    # Extend the original dictionary
    tbc_nodes_dict['place_to_transitions'] = place_to_transitions

    return tbc_nodes_dict

# used
def find_nets_with_net(net_in, t_add, log, pre_post_dict, skip, loop, old=False):
    net = copy.deepcopy(net_in)
    im, fm = find_im_fm(net)
    net, im, fm = convert_start_end_tau_to_silent(net, im, fm)
    net = sync_transition_names_with_labels(net)
    new_net = adjust_net_for_SynthesisMiner(net)
    im, fm = find_im_fm(new_net)
    if not check_wfnet(new_net):
        print('Subnet not wf-net')
    # SC Net
    new_net, new_im, new_fm = toggle_sct(net=new_net, initial_marking=im)
    tbc_nodes_inds_dict_old = get_tbc_nodes_inds_dict(net=new_net, new_t_name=t_add, log=log, noise_threshold=0)
    # add place_to_transition:
    tbc_nodes_inds_dict_old = extend_tbc_nodes_inds_dict(tbc_nodes_inds_dict_old, new_net, t_add, pre_post_dict)

    tbc_nodes_inds_dict = get_tbc_nodes_new_dict(net=new_net, new_t_name=t_add, pre_post_dict=pre_post_dict, log=log)
    n_tbc_nodes = len(tbc_nodes_inds_dict['transitions']) + len(tbc_nodes_inds_dict['places'])
    n_tbc_nodes_old = len(tbc_nodes_inds_dict_old['transitions']) + len(tbc_nodes_inds_dict_old['places'])
    # create a dict for returning s.t. it works with results_of_iteration
    node_count_dict = {}
    if n_tbc_nodes > n_tbc_nodes_old:
        # use old node dict
        print("Using old node dict")
        print("Old: ", tbc_nodes_inds_dict_old)
        print("New: ", tbc_nodes_inds_dict)
        print("Net: ", net)
        tbc_nodes_inds_dict = tbc_nodes_inds_dict_old
    print("Nodes: ", n_tbc_nodes)
    print("Nodes old: ", n_tbc_nodes_old)
    if old:
        node_count_dict['n_nodes_tran'] = n_tbc_nodes_old
    else:
        node_count_dict['n_nodes_tran'] = n_tbc_nodes

    tbc_nodes_inds_dict_combined = copy.deepcopy(tbc_nodes_inds_dict)
    tbc_nodes_inds_dict_combined = place_fitness_old(net, im, fm, tbc_nodes_inds_dict_combined, log)

    if not old:
        print('Finding new nets (transitions)')
        t_before = time.perf_counter()
        net_list = get_all_new_nets_faster(net=new_net, new_t_name=t_add, tbc_nodes_inds_dict=tbc_nodes_inds_dict,
                                           use_new_rule=False, use_skip=skip, use_loop=loop)
        t_after = time.perf_counter()
        t_diff_1 = t_after - t_before
        print('Time for finding nets: ', t_diff_1)
        node_count_dict['time_tran'] = t_diff_1
        node_count_dict['n_nets_tran'] = len(net_list)

    if old:
        t_before = time.perf_counter()
        net_list = get_all_new_nets_faster(net=new_net, new_t_name=t_add,
                                               tbc_nodes_inds_dict=tbc_nodes_inds_dict_old,
                                               use_new_rule=False, use_skip=skip, use_loop=loop)
        t_after = time.perf_counter()
        t_diff_1 = t_after - t_before
        print('Time for finding nets: ', t_diff_1)
        node_count_dict['time_tran'] = t_diff_1
        node_count_dict['n_nets_tran'] = len(net_list)

    if len(net_list) == 0:
        print("No net returned (calculated)")
    print(node_count_dict)
    return net_list, node_count_dict

# used
def get_all_new_nets_faster(net, new_t_name: str, tbc_nodes_inds_dict=None, use_new_rule=False, use_skip=False,
                            use_loop=False):
    m = incidence_matrix.construct(net)
    mat = np.array(m.a_matrix)
    trans_dict = {k.name: int(v) for k, v in m.transitions.items()}
    places_dict = {k.name: int(v) for k, v in m.places.items()}
    source_ind = places_dict['source']
    sink_ind = places_dict['sink']
    start_tau_ind = trans_dict['start_tau']
    end_tau_ind = trans_dict['end_tau']
    short_circuit_ind = trans_dict['short_circuited_transition']

    new_trans = get_new_nodes_gen_by_LD_rules(mat, source_ind=source_ind, sink_ind=sink_ind,
                                              start_tau_ind=start_tau_ind,
                                              end_tau_ind=end_tau_ind, short_circuit_ind=short_circuit_ind,
                                              node_type='t',
                                              tbc_nodes_inds_dict=tbc_nodes_inds_dict)

    new_places = get_new_nodes_gen_by_LD_rules(mat, source_ind=source_ind, sink_ind=sink_ind,
                                               start_tau_ind=start_tau_ind,
                                               end_tau_ind=end_tau_ind, short_circuit_ind=short_circuit_ind,
                                               node_type='p',
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
    # print('Places', list(new_places))
    # print('Transitions', list(new_trans))
    # print('Abs pairs: ', list(possible_abs_pairs_gen))

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
        for new_t_in_loop, new_t_skippable in product([False, True], repeat=2):
            # Skip the iteration if use_skip is False and new_t_skippable is True
            if not use_skip and new_t_skippable:
                continue
            # Skip the iteration if use_loop is False and new_t_in_loop is True
            if not use_loop and new_t_in_loop:
                continue

            updated_incidence_mat, updated_places_dict, updated_trans_dict = apply_LD_p_then_abs_rules(
                incidence_mat=mat,
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
        for new_t_in_loop, new_t_skippable in product([False, True], repeat=2):
            # Skip the iteration if use_skip is False and new_t_skippable is True
            if not use_skip and new_t_skippable:
                continue
            # Skip the iteration if use_loop is False and new_t_in_loop is True
            if not use_loop and new_t_in_loop:
                continue
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
        for new_t_in_loop, new_t_skippable in product([False, True], repeat=2):
            # Skip the iteration if use_skip is False and new_t_skippable is True
            if not use_skip and new_t_skippable:
                continue
            # Skip the iteration if use_loop is False and new_t_in_loop is True
            if not use_loop and new_t_in_loop:
                continue
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

    if len(new_nets_list) == 0:
        print('No new nets found')
    return new_nets_list


def extend_petri_net_w_ld(net: PetriNet, im, fm, log, t_add, old=False):
    # Function to add the transition t_add to the net
    # 1. Find prev and foll transitions
    prev, foll, skip, loop = find_prev_foll_skip_loop(log, t_add, net)
    print('Prev: ', prev)
    print('Foll: ', foll)
    print('Skip: ', skip)
    print('Loop: ', loop)

    pre_post_dict = {
        'pre': prev,
        'post': foll
    }

    # all is both filters and approximation all used
    results_of_iteration = {
        't_add': t_add, 'prev': prev, 'foll': foll, 'skip': skip, 'loop': loop, 'net_trans': len(net.transitions),
        'net_places': len(net.places), 'net_nodes': len(net.transitions) + len(net.places)}


    # from utils.place_analysis import place_fitness, place_fitness_old
    # place_fitness(net, pre_post_dict, log, 0.95)

    print('Finding nets (standard)')
    new_nets, nodes_count_dict = find_nets_with_net(net, t_add, log, pre_post_dict, skip, loop, old)
    results_of_iteration['n_nets_found'] = nodes_count_dict['n_nets_tran']
    results_of_iteration['n_nodes_used'] = nodes_count_dict['n_nodes_tran']
    results_of_iteration['time_finding_nets'] = nodes_count_dict['time_tran']

    if len(new_nets) == 0 and len(new_nets) == 0:
        print('No new nets found')
        return net, im, fm, None, None, None, None,

    best_f1 = 0
    best_precision = 0
    best_fitness = 0
    best_net = net, im, fm
    print(f'we got {len(new_nets)} new nets')
    i = 0
    count_not_sound = 0
    # Go through new_nets, find best based on F1-score
    last_n = None
    print('Finding best net')
    t_conf_start = time.perf_counter()
    for net_dict in new_nets:
        i += 1
        # print(i)
        new_petri_dict = create_pn_from_incidence_mat(incidence_mat=net_dict['incidence_mat'],
                                                      places_dict=net_dict['places_dict'],
                                                      trans_dict=net_dict['trans_dict'],
                                                      return_net_dict=True)
        used_rule = net_dict['rule']
        print('Used rule: ', used_rule)
        new_petri_temp = new_petri_dict['petri']
        new_petri = copy.deepcopy(new_petri_temp)
        # Remove 'short_circuit_transition'
        new_petri = remove_tran_by_name(new_petri, trans_name='short_circuited_transition')
        new_petri_copy = copy.deepcopy(new_petri)
        im2, fm2 = find_im_fm(new_petri_copy)
        if check_wfnet(new_petri_copy):
            # pm4py.view_petri_net(new_petri_copy, format='svg')
            pm4py.save_vis_petri_net(new_petri_copy, im2, fm2,
                                     os.path.join('results', 'nets', 'temp', 'net_' + str(i) + '.png'))
            if not check_soundness.check_easy_soundness_net_in_fin_marking(new_petri_copy, im2, fm2):
                print('Net not sound')
                count_not_sound += 1
            else:
                fitness = replay_fitness_evaluator.apply(log, new_petri_copy, im2, fm2)
                fitness = fitness['log_fitness']
                precision = pm4py.precision_alignments(log, new_petri_copy, im2, fm2)
                if fitness != 0:
                    f1 = 2 * ((precision * fitness) / (precision + fitness))
                else:
                    f1 = 0
                print(precision, fitness, f1)
                if f1 > best_f1:
                    best_f1 = f1
                    best_precision = precision
                    best_fitness = fitness
                    best_net = new_petri_copy, im2, fm2
                    last_n = i
                    best_rule = used_rule
                if best_f1 == 1:
                    print('Conformance checking on subnet already finished here', last_n)
                    break
        else:
            print('Net not wf-net')
            path = os.path.join('results', 'nets', 'temp', 'not_wf_net_' + str(i) + '.png')
            pm4py.save_vis_petri_net(new_petri_copy, im2, fm2, path)

    print()
    t_conf_end = time.perf_counter()
    t_diff_conf = t_conf_end - t_conf_start
    print('Conformance checking: ', t_diff_conf)
    results_of_iteration['time_conf_tran'] = t_diff_conf
    # print('Net with best f1', best_f1)
    print('Best F1 score', best_f1)
    results_of_iteration['f1_tran'] = best_f1
    results_of_iteration['precision_tran'] = best_precision
    results_of_iteration['fitness_tran'] = best_fitness

    return best_net[0], best_net[1], best_net[2], best_f1, results_of_iteration


def one_iteration(net, im, fm, importlog, testnode=None, tran_tracking_l=None, old=False):
    # 1. step, detect activity to cut (predefined in experiment)
    if testnode is None:
        raise ValueError('No testnode given')
    else:
        cut_act = testnode

    # 2. Step, remove detected activity, get reduced net
    print('Step 2:')
    red_net, red_im, red_fm = apply_reduction(cut_act, net, im, fm)

    for f in os.listdir(os.path.join('results', 'nets', 'temp')):
        os.remove(os.path.join('results', 'nets', 'temp', f))
    red_im, red_fm = find_im_fm(red_net)
    pm4py.save_vis_petri_net(red_net, red_im, red_fm, os.path.join('results', 'nets', 'temp', 'red_net.png'))

    if not check_soundness.check_easy_soundness_of_wfnet(red_net):
        print('Net not sound!')
        red_im, red_fm = find_im_fm(red_net)
        pm4py.view_petri_net(red_net, red_im, red_fm)
        return net, im, fm, 0, None, None

    # 3. Step, add the detected activity
    print('Step 3:')
    more_red_net, more_red_im, more_red_fm = reduce_net(red_net)
    pm4py.save_vis_petri_net(more_red_net, more_red_im, more_red_fm,
                             os.path.join('results', 'nets', 'temp', 'more_red_net.png'))

    fix_net, fix_im, fix_fm, f1, results_of_iteration = extend_petri_net_w_ld(
        more_red_net, more_red_im, more_red_fm, importlog, cut_act, old)

    return fix_net, fix_im, fix_fm, f1, results_of_iteration, cut_act


def export_results(results, filename, threshold, old):
    df = pd.DataFrame.from_dict(results, orient='index')
    # sort df on t_add alphabetically ascending
    df = df.sort_values(by='t_add')

    threshold = str(threshold).replace('.', '_')
    if old:
        filename = 'results_' + filename.split('.')[0] + threshold + '_old.csv'
    else:
        filename = 'results_' + filename.split('.')[0] + threshold + '.csv'
    resultfile = os.path.join('results', 'analysis', 'ex_place', filename)
    print(df)

    df.to_csv(resultfile)


if __name__ == "__main__":
    run = sys.argv[1]
    base_dir = os.path.join(os.getcwd(), 'logs')
    # return
    if run == 'test':
        filename = 'demo_c_f.xes'
        pt = create_model_pt()
        net, im, fm = convert_to_petri_net(pt)
        init_threshold = 0
    elif run == '17A_2':
        filename = 'BPIC2017_A.xes.gz'
        net, im, fm = pm4py.read_pnml('models/BPIC_2017_A_02.pnml')
        init_threshold = 0.2
    elif run == '17A_4':
        filename = 'BPIC2017_A.xes.gz'
        net, im, fm = pm4py.read_pnml('models/BPIC_2017_A_04.pnml')
        init_threshold = 0.4
    elif run == '17O_2':
        filename = 'BPIC2017_O.xes.gz'
        net, im, fm = pm4py.read_pnml('models/BPIC_2017_O_02.pnml')
        init_threshold = 0.2
    elif run == '17O_4':
        filename = 'BPIC2017_O.xes.gz'
        net, im, fm = pm4py.read_pnml('models/BPIC_2017_O_04.pnml')
        init_threshold = 0.4
    elif run == 'help_2':
        filename = 'helpdesk_with_end.xes'
        net, im, fm = pm4py.read_pnml('models/helpdesk_02.pnml')
        init_threshold = 0.2
    elif run == 'help_4':
        filename = 'helpdesk_with_end.xes'
        net, im, fm = pm4py.read_pnml('models/helpdesk_04.pnml')
        init_threshold = 0.4
    else:
        raise ValueError('Invalid run parameter')

    file_path = os.path.join(base_dir, filename)
    importlog = xes_importer.apply(file_path)

    variant_count = pm4py.get_variants(log=importlog)
    print('Number of variants: ', len(variant_count))
    print('Number of traces: ', len(importlog))
    events = pm4py.get_event_attribute_values(importlog, 'concept:name')
    print('Number of events: ', len(events))

    all_one = True
    threshold = 0
    if isinstance(init_threshold, int):
        threshold = str(init_threshold)
    elif isinstance(init_threshold, float):
        threshold = str(init_threshold).split('.')[1]

    # use old for old (original) method in same setting
    if len(sys.argv) > 2:
        if sys.argv[2] == 'old':
            old = True
        else:
            raise ValueError('Invalid parameter')
    else:
        old = False

    init_net_name = os.path.join('results', 'nets',
                                 filename.split('.')[0] + '_' + threshold + '_init_net.svg')

    pm4py.save_vis_petri_net(net, im, fm, init_net_name)

    precision = pm4py.precision_alignments(importlog, net, im, fm)

    fitness_dict = replay_fitness_evaluator.apply(importlog, net, im, fm,
                                                  variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)
    recall = fitness_dict['percentage_of_fitting_traces'] / 100
    fitness = fitness_dict['log_fitness']
    # time_f1 = {}
    if recall != 0:
        f1 = 2 * ((precision * recall) / (precision + recall))
    elif fitness != 0:
        f1 = 2 * ((precision * fitness) / (precision + fitness))
        print('fitness used for recall')
    print('Initial precision, recall, fitness, f1: ', precision, recall, fitness, f1)
    init_f1 = f1
    prev_f1 = f1
    # results dict: key = step, value = [complete time, f1, step_times, cut_act]
    results_0 = {
        'f1_tran': f1,
        'precision_tran': precision,
        'fitness_tran': fitness
    }
    results = {0: results_0}
    no_improvement_count = 0
    for t in net.transitions:
        if t.label is not None and str(t.label) not in ['End', 'Closed', 'end']:
            print(t.label)
    cut_act_dict = {}

    # go through each transition
    all_act = set()
    i = 0

    if os.path.exists('places.csv'):
        os.remove('places.csv')

    for t in net.transitions:
        if t.label is not None and str(t.label) not in ['End', 'Closed', 'end']:
            net_copy = copy.deepcopy(net)
            new_im, new_fm = find_im_fm(net_copy)
            print()
            print(t.label)
            # todo
            new_net_temp, new_im_temp, new_fm_temp, f1, results_of_iteration, cut_act = one_iteration(
                net_copy, new_im, new_im, importlog, t.label, old=old)
            cut_act_dict[i] = results_of_iteration
            i += 1
            # pm4py.view_petri_net(new_net_temp, new_im_temp, new_fm_temp, format='svg')
            pm4py.save_vis_petri_net(new_net_temp, new_im_temp, new_fm_temp,
                                     file_path=os.path.join('results', 'nets', 'ex_place',
                                                            filename.split('.')[0] + '_' + threshold + '_' + str(t.label) + '.svg'))
            place_filename = 'place_results_' + filename.split('.')[0] + threshold + '.csv'

            places_df = pd.read_csv('places.csv')
            places_df.to_csv(os.path.join('results', 'analysis', 'ex_place', place_filename))

            export_results(cut_act_dict, filename, threshold, old)
    print('All single iterations done')
    sys.exit()
