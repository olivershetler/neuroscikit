import os, sys
import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.study_space import Session
from _prototypes.unit_matcher.unit import jensen_shannon_distance, spike_level_feature_array
from core.spikes import SpikeCluster
from library.batch_space import SpikeClusterBatch


def compute_distances(session1_cluster: SpikeClusterBatch, session2_cluster: SpikeClusterBatch, JSD=True, MSDoWFM=False): # change to feature vector array
    """
    Iterates through all the across-session unit pairings and computing their respective Jensen-Shannon distances
    Parameters
    ----------
    session1_cluster: SpikeClusterBatch
    session2_cluster: SpikeClusterBatch
    JSD: bool
        If True, computes the Jensen-Shannon distance between the two sessions' unit feature vectors
    MSDoWFM: bool
        If True, computes the Mean Squared Difference of Waveform Means between the two sessions' unit feature vectors
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

            distance = jensen_shannon_distance(session1_feature_arrays[i], session2_feature_arrays[j])
            print('JSD: ' + str(distance))

            if 'JSD' not in session1_unit_clusters[i].stats_dict:
                session1_unit_clusters[i].stats_dict['JSD'] = []
            session1_unit_clusters[i].stats_dict['JSD'] = distance

            if 'JSD' not in session2_unit_clusters[j].stats_dict:
                session2_unit_clusters[j].stats_dict['JSD'] = []
            session2_unit_clusters[j].stats_dict['JSD'] = distance

            distances[i,j] = distance
            pairs[i,j] = [session1_unit_clusters[i].cluster_label, session2_unit_clusters[j].cluster_label]

    return distances, pairs

def extract_full_matches(distances, pairs):
    full_matches = []
    full_match_distances = []
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
                full_match_distances.append(distances[i,j])
                row_mask[i] = False
                col_mask[j] = False
                assert sorted(unit1_pairs[unit1_min]) == sorted(unit2_pairs[unit2_min])

    remaining_distances = distances[row_mask,:][:,col_mask]
    remaining_pairs = pairs[row_mask,:][:,col_mask]

    return full_matches, full_match_distances, remaining_distances, remaining_pairs


def guess_remaining_matches(distances, pairs):
    row_ind, col_ind = linear_sum_assignment(distances)
    assert len(row_ind) == len(col_ind)
    remaining_matches = []
    remaining_match_distances = []
    for i in range(len(row_ind)):
        unit_pair = pairs[row_ind[i], col_ind[i]]
        remaining_matches.append(unit_pair)
        remaining_match_distances.append(distances[row_ind[i], col_ind[i]])

    # session1_unmatched = list(set(np.arange(len(distances))) - set(remaining_matches[0]))
    if len(distances) > 0:
        session2_unmatched = list(set(list(np.arange(len(distances[0])))) - set(list(col_ind)))
        session1_unmatched = list(set(list(np.arange(len(distances)))) - set(list(row_ind)))
    else:
        session2_unmatched = []
        session1_unmatched = []

    unmatched_2 = []
    for i in range(len(session2_unmatched)):
        # save the actual cell label of the unmatched cell
        # get the column of that cell (list of pairings with actual cell labels of session 1 and session 2 cells)
        # take any row from that column, the second number in the pair at (row,col) is the true cell label for the cell in session 2
        # make sure any pair in that column has the same second number [session1 id, session2 id]
        unit_id = pairs[0, session2_unmatched[i]][-1]
        # assert type(unit_id) == float or type(unit_id) == np.float64 or type(unit_id) == np.int64 or type(unit_id) == int or type(unit_id) == np.int32 or type(unit_id) == np.float32
        # assert unit_id == pairs[1, session2_unmatched[j]][-1]
        # unmatched_2.append([0, unit_id])
        unmatched_2.append(unit_id)

    unmatched_1 = []
    for i in range(len(session1_unmatched)):
        unit_id = pairs[session1_unmatched[i], 0][0]
        # unmatched_1.append([0, unit_id])
        unmatched_1.append(unit_id)

    return remaining_matches, remaining_match_distances, unmatched_2, unmatched_1

def compare_sessions(session1: Session, session2: Session):
    """
    FD = feature dict
    1 & 2 = sessions 1 & 2 (session 2 follows session 1)
    """
    # compare output of extract features from session1 and session2
    # return mapping dict from session2 old label to new matched label based on session1 cell labels

    distances, pairs = compute_distances(session1.get_spike_data()['spike_cluster'], session2.get_spike_data()['spike_cluster'])
    full_matches, full_match_distances, remaining_distances, remaining_pairs = extract_full_matches(distances, pairs)

    remaining_matches, remaining_match_distances, unmatched_2, unmatched_1 = guess_remaining_matches(remaining_distances, remaining_pairs)

    # to_stack = []
    # if np.asarray(full_matches).size > 0:
    #     to_stack.append(full_matches)
    # if np.asarray(remaining_matches).size > 0:
    #     to_stack.append(remaining_matches)
    # if np.asarray(unmatched).size > 0:
    #     to_stack.append(unmatched)

    # matches = to_stack[0]
    # for i in range(1,len(to_stack)):
    #     matches = np.vstack((matches, to_stack[i]))

    if np.asarray(full_matches).size > 0 and np.asarray(remaining_matches).size > 0:
        matches = np.vstack((full_matches, remaining_matches))
        match_distances = np.hstack((full_match_distances, remaining_match_distances))
    else:
        matches = full_matches
        match_distances = full_match_distances

    # matches = np.vstack((full_matches, remaining_matches))
    # matches = np.vstack((matches, unmatched))

    return matches, match_distances, unmatched_2, unmatched_1



