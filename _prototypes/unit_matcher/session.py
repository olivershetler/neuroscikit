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
    # mask = np.ones(distances.shape, bool)
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

    # for i in range(len(row_mask)):


    distances = distances[row_mask,:][:,col_mask]
    # distances = distances[:, col_mask]
    pairs = pairs[row_mask,:][:,col_mask]
    # pairs = pairs[:, col_mask]
    # pairs = pairs[row_mask, col_mask]
    # distances = distances[idx[0], idx[1]]
    # pairs = pairs[idx]

    return full_matches, distances, pairs


    # return matches

def guess_remaining_matches(distances, pairs):
    # smaller side of bipartite graph (pop everythign that is a match till only no matches left)
    # hungarian algorithm (scipy)
    # return additiona list of matches
    # return any unmatched units/leftover units + sessions
    # cchange return output of compare_sessions()
    row_ind, col_ind = linear_sum_assignment(distances)
    remaining_matches = pairs[row_ind,col_ind]
    # session1_unmatched = list(set(np.arange(len(distances))) - set(remaining_matches[0]))
    session2_unmatched = list(set(list(np.arange(len(distances[0])))) - set(list(col_ind)))

    leftover_units = []

    # for i in range(len(session1_unmatched)):
    #     unit_id = pairs[session1_unmatched[i],:][0]
    #     leftover_units.append([unit_id, 0])

    # ONLY CARE ABOUT SESSION 2 UNMATCHED CELLS, SESSION 1 UNMATCHED DO NOT CHANGE LABEL
    for j in range(len(session2_unmatched)):
        unit_id = pairs[:, session2_unmatched[j]][0]
        leftover_units.append([0, unit_id])
    
    return np.array(remaining_matches).T, leftover_units

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
    
    remaining_matches, unmmatched = guess_remaining_matches(remaining_distances, remaining_pairs)
    print(full_matches, remaining_matches, unmmatched)
    matches = np.vstack((full_matches, remaining_matches))
    matches = np.vstack((matches, unmmatched))
    
    map_dict = map_unit_matches(matches)

    return map_dict 



