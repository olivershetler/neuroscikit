import os, sys
import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.study_space import Session
from _prototypes.unit_matcher.unit import jensen_shannon_distance, spike_level_feature_array
from core.spikes import SpikeCluster
from library.batch_space import SpikeClusterBatch


def compute_distances(session1_cluster: SpikeClusterBatch, session2_cluster: SpikeClusterBatch): # change to feature vector array
    """
    Iterates through all the across-session unit pairings and computing their respective Jensen-Shannon distances
    """

    session1_unit_clusters = session1_cluster.get_spike_cluster_instances()
    session2_unit_clusters = session2_cluster.get_spike_cluster_instances()

    distances = np.zeros((len(session1_unit_clusters), len(session2_unit_clusters)))
    pairs = np.zeros((len(session1_unit_clusters), len(session2_unit_clusters), 2))

    session1_feature_arrays = []
    session2_feature_arrays = []
    for i in range(len(session1_unit_clusters)):
        session1_feature_arrays.append(spike_level_feature_array(session1_unit_clusters[i], 1/session1_cluster.sample_rate))
    for j in range(len(session2_unit_clusters)):
        session2_feature_arrays.append(spike_level_feature_array(session2_unit_clusters[j], 1/session2_cluster.sample_rate))

    for i in range(len(session1_feature_arrays)):
        for j in range(len(session2_feature_arrays)):
            print('Session1 ' + str(i) + '; Session2 ' + str(j))

            # idx1 = np.where(session1_feature_array != session1_feature_array)[0]
            # idx2 = np.where(session2_feature_array != session2_feature_array)[0]
            # print(idx1, idx2)
            # assert len(idx1) == 0
            # assert len(idx2) == 0

            distance = jensen_shannon_distance(session1_feature_arrays[i], session2_feature_arrays[j])

            # distances.append(distance)
            # pairs.append[[unit1, unit2]]

            distances[i,j] = distance
            pairs[i,j] = [session1_unit_clusters[i].cluster_label, session2_unit_clusters[j].cluster_label]

    return distances, pairs

def extract_full_matches(distances, pairs):
    full_matches = []
    row_mask = np.ones(distances.shape[0], bool)
    col_mask = np.ones(distances.shape[1], bool)


    for i in range(distances.shape[0]):
        unit1_pairs = pairs[i]
        unit1_min = np.argmin(distances[i,:])
        for j in range(distances.shape[1]):
            unit2_pairs = pairs[:,j]
            unit2_min = np.argmin(distances[:,j])
            if unit1_min == j and unit2_min == i:
                full_matches.append(unit1_pairs[unit1_min])
                row_mask[i] = False
                col_mask[j] = False
                assert sorted(unit1_pairs[unit1_min]) == sorted(unit2_pairs[unit2_min])

    distances = distances[row_mask,:][:,col_mask]
    pairs = pairs[row_mask,:][:,col_mask]

    return full_matches, distances, pairs


def guess_remaining_matches(distances, pairs):
    row_ind, col_ind = linear_sum_assignment(distances)
    assert len(row_ind) == len(col_ind)
    remaining_matches = []
    for i in range(len(row_ind)):
        unit_pair = pairs[row_ind[i], col_ind[i]]
        remaining_matches.append(unit_pair)

    # session1_unmatched = list(set(np.arange(len(distances))) - set(remaining_matches[0]))
    session2_unmatched = list(set(list(np.arange(len(distances[0])))) - set(list(col_ind)))

    leftover_units = []
    # ONLY CARE ABOUT SESSION 2 UNMATCHED CELLS, SESSION 1 UNMATCHED DO NOT CHANGE LABEL
    for j in range(len(session2_unmatched)):
        # save the actual cell label of the unmatched cell
        # get the column of that cell (list of pairings with actual cell labels of session 1 and session 2 cells)
        # take any row from that column, the second number in the pair at (row,col) is the true cell label for the cell in session 2
        # make sure any pair in that column has the same second number [session1 id, session2 id]
        unit_id = pairs[0, session2_unmatched[j]][-1]
        assert type(unit_id) == float or type(unit_id) == np.float64 or type(unit_id) == np.int64 or type(unit_id) == int or type(unit_id) == np.int32 or type(unit_id) == np.float32
        assert unit_id == pairs[1, session2_unmatched[j]][-1]
        leftover_units.append([0, unit_id])
    
    return np.array(remaining_matches), leftover_units

def map_unit_matches(matches):
    map_dict = {}
    
    for pair in matches:
        map_dict[int(pair[1])] = int(pair[0])

    return map_dict

def compare_sessions(session1: Session, session2: Session):
    """
    FD = feature dict
    1 & 2 = sessions 1 & 2 (session 2 follows session 1)
    """
    # compare output of extract features from session1 and session2
    # return mapping dict from session2 old label to new matched label based on session1 cell labels

    distances, pairs = compute_distances(session1.get_spike_data()['spike_cluster'], session2.get_spike_data()['spike_cluster'])
    full_matches, remaining_distances, remaining_pairs = extract_full_matches(distances, pairs)
    remaining_matches, unmatched = guess_remaining_matches(remaining_distances, remaining_pairs)

    to_stack = []
    if np.asarray(full_matches).size > 0:
        to_stack.append(full_matches)
    if np.asarray(remaining_matches).size > 0:
        to_stack.append(remaining_matches)
    if np.asarray(unmatched).size > 0:
        to_stack.append(unmatched)
    
    matches = to_stack[0]
    for i in range(1,len(to_stack)):
        matches = np.vstack((matches, to_stack[i]))

    # matches = np.vstack((full_matches, remaining_matches))
    # matches = np.vstack((matches, unmatched))
    
    map_dict = map_unit_matches(matches)

    return map_dict 



