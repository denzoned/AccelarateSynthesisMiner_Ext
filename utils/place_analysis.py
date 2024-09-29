from collections import defaultdict
from enum import Enum

import numpy as np
import pandas as pd
# from pm4py import PetriNet, Marking
# from pm4py.algo.conformance.alignments.decomposed import algorithm as decomp_alignments
# from pm4py import conformance_diagnostics_token_based_replay as token_replay_factory
# from pm4py.objects.petri_net.utils import incidence_matrix
# from pm4py.util import constants, exec_utils
from pm4py.statistics.variants.log import get as variants_module
# from pm4py.algo.conformance.tokenreplay.diagnostics import root_cause_analysis
#
# from testing.clean_unblockIM import toggle_sct


# class Parameters(Enum):
#     ACTIVITY_KEY = constants.PARAMETER_CONSTANT_ACTIVITY_KEY

# def find_matching_place(net, trans_info):
#     # Attempt to find a matching place in the target_net based on connected transitions
#     for place in net.places:
#         incoming_transitions = set([arc.source.name for arc in place.in_arcs])
#         outgoing_transitions = set([arc.target.name for arc in place.out_arcs])
#
#         if incoming_transitions == set(trans_info['incoming']) and outgoing_transitions == set(trans_info['outgoing']):
#             return place.name
#     return None

def find_transition(net, transition_name):
    print(net.transitions)
    for transition in net.transitions:
        if transition.label == transition_name:
            return transition
    return None

def place_fitness(net, pre_post_dict, log, threshold=0.95):
    # pre_post_dict includes the transition names of the pre and post sets
    print('pre_post_dict: ', pre_post_dict)
    eval_places = {}
    # use a dictionary to save the pre transitions
    pre_dict = {}

    for pre_tran_name in pre_post_dict['pre']:
        pre_dict[pre_tran_name] = set()
        # find the transition in the net
        transition = find_transition(net, pre_tran_name)
        # find the places connected to the transition
        for arc in transition.out_arcs:
            pre_dict[pre_tran_name].add(arc.target.name)
    print('pre_dict: ', pre_dict)

    post_dict = {}

    for post_tran_name in pre_post_dict['post']:
        post_dict[post_tran_name] = set()
        # find the transition in the net
        transition = find_transition(net, post_tran_name)
        # find the places connected to the transition
        for arc in transition.in_arcs:
            post_dict[post_tran_name].add(arc.source.name)

    print('post_dict: ', post_dict)
    print()

    # now we check the fitness of the places




def place_fitness_old(net, initial_marking, final_marking, tbc_dict, log, threshold=0.95):
    places_info = tbc_dict['place_to_transitions']
    print('places_info: ', places_info)
    for place in places_info.keys():
        places_info[place].update({
            "underfed": 0,
            "overfed": 0,
            "fitting": 0,
            "total_traces": 0,
            "activated_traces": 0,
            "freq_underfed": 0,
            "freq_overfed": 0,
            "freq_fitting": 1,
            "rel_underfed": 0,
            "rel_overfed": 0,
            "rel_fitting": 1,
            "tau_rel_underfed": False,
            "tau_rel_overfed": False,
            "tau_rel_fitting": True,
            "tau_freq_underfed": False,
            "tau_freq_overfed": False,
            "tau_freq_fitting": True,
            "remove": False,
            "Iteration": 0
        })
    # Get variants from the log
    variants = variants_module.get_variants(log)
    silent_transitions = set(t.name for t in net.transitions if t.label is None)

    for variant, traces in variants.items():
        count = len(traces)

        # Initialize tokens needed and tokens produced for each place
        tokens_needed = defaultdict(int)
        tokens_produced = defaultdict(int)

        for place, info in places_info.items():
            activated = False  # Flag to check if the place is activated in the current trace

            for transition in variant:
                if transition in info['incoming'] or transition in info['outgoing']:
                    activated = True  # The place is activated if any of its connected transitions appear in the trace
                    if transition in info['outgoing']:
                        tokens_needed[place] += 1
                    if transition in info['incoming']:
                        tokens_produced[place] += 1

            # Update the activated traces count
            if activated:
                # place_fitness_dict[place]['activated_traces'] += count
                places_info[place]['activated_traces'] += count

            # Handle silent transitions
            if any(t in silent_transitions for t in info['outgoing']) or any(str(t).startswith("tau") for t in info['outgoing']):
                if tokens_needed[place] < tokens_produced[place]:  # Only consume a token if there's one left
                    tokens_needed[place] += 1

            if any(t in silent_transitions for t in info['incoming']) or any(str(t).startswith("tau") for t in info['incoming']):
                if tokens_produced[place] < tokens_needed[place]:  # Only produce a token if one is missing
                    tokens_produced[place] += 1

        for place in places_info.keys():
            needed = tokens_needed[place]
            produced = tokens_produced[place]

            places_info[place]['total_traces'] += count
            # place_fitness_dict[place]['total_traces'] += count

            if needed < produced:
                # place_fitness_dict[place]['underfed'] += count
                places_info[place]['underfed'] += count
            elif needed > produced:
                # place_fitness_dict[place]['overfed'] += count
                places_info[place]['overfed'] += count
            else:
                # place_fitness_dict[place]['fitting'] += count
                places_info[place]['fitting'] += count

        # Check if each place is underfed, overfed, or fitting
        # for place, metrics in place_fitness_dict.items():
    for place, metrics in places_info.items():
        total = len(log)
        activated = metrics['activated_traces']

        if total > 0:
            metrics['freq_underfed'] = metrics['underfed'] / total
            metrics['freq_overfed'] = metrics['overfed'] / total
            metrics['freq_fitting'] = metrics['fitting'] / total

            if activated > 0:
                metrics['rel_underfed'] = metrics['underfed'] / activated
                metrics['rel_overfed'] = metrics['overfed'] / activated
                metrics['rel_fitting'] = 1 - (metrics['rel_underfed'] + metrics['rel_overfed'])

                metrics['tau_rel_underfed'] = metrics['rel_underfed'] > (1 - threshold)
                metrics['tau_rel_overfed'] = metrics['rel_overfed'] > (1 - threshold)
                metrics['tau_rel_fitting'] = metrics['rel_fitting'] >= threshold

            metrics['tau_freq_underfed'] = metrics['freq_underfed'] > (1 - threshold)
            metrics['tau_freq_overfed'] = metrics['freq_overfed'] > (1 - threshold)
            metrics['tau_freq_fitting'] = 1 - (metrics['freq_underfed'] + metrics['freq_overfed']) >= threshold

        if (metrics['tau_rel_underfed']) and ('output_of_pre' in metrics['place_type']):
            metrics['remove'] = True
            # print('Place should be removed because it is underfed: ', place)
        if (metrics['tau_rel_overfed']) and ('input_of_post' in metrics['place_type']):
            metrics['remove'] = True
            # print('Place should be removed because it is overfed: ',  place)
        if (metrics['tau_rel_overfed'] or metrics['tau_rel_underfed']) and ('normal' in metrics['place_type']):
            metrics['remove'] = True
            # print('Place should be removed because it is not fitting: ', place)
    # if place_type is not given, remove the place if it is not fitting

    print('Given places: ', tbc_dict['places'])
    for place in tbc_dict['places']:
        if places_info[place]['remove']:
            tbc_dict['places'].remove(place)
            print('Removed place: ', place)

    places_df = pd.DataFrame.from_dict(places_info, orient='index')

    try:
        places_df_old = pd.read_csv('places.csv')
        max_iter = places_df_old['Iteration'].max()
        places_df['Iteration'] = max_iter + 1

        places_df = pd.concat([places_df_old, places_df], axis=0)
    except:
        pass

    places_df.to_csv('places.csv', index=False)

    return tbc_dict

