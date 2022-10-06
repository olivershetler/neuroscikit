import os, sys
import numpy as np

from x_io.rw.axona.batch_read import make_study

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.study_space import Session
# from x_io.rw.axona.batch_read import _read_cut
from _prototypes.unit_matcher.write_axona import get_current_cut_data, write_cut, format_new_cut_file_name, apply_remapping
from _prototypes.unit_matcher.read_axona import temp_read_cut

"""

This module reads axona cut and tetrode files. DONE (batch_read in x_io or single pair read in read_axona)
It then extracts the spike waveforms from the cut file. DONE (batch_read or single pair read in read_axona)
It then matches the spike waveforms to the units in the tetrode file. DONE (part of core class loading)
It then produces a dictionary of the spike waveforms for each unit. DONE (No dictionary --> Core class with waveforms per Spike. Collection Spike = Cluster)
It then extracts features from the spike waveforms for each unit. 
It then matches the spike waveforms to the units in the tetrode file.
It then produces a remapping of the units in the tetrode file. 
It then applies the remapping to the cut file data. DONE (map dict changes cut data)
It then writes the remapped cut file data to a new cut file. DONE (new cut data writs to file)
Read, Retain, Map, Write
"""


def run_unit_matcher(directory: list, settings: dict):
    # make study --> will load + sort data: SpikeClusterBatch (many units) --> SpikeCluster (a unit) --> Spike (an event)
    study = make_study(directory, settings)
    # make animals
    study.make_animals()
    for animal in study.animals:
        # SESSIONS INSIDE OF ANIMAL WILL BE SORTED SEQUENTIALLY AS PART OF ANIMAL(WORKSPACE) CLASS IN STUDY_SPACE.PY
        prev = None 
        curr = None
        for session in animal.sessions:
            # if first session of sequence there is no prev session
            if prev != None:
                prev = curr
            # update refernece of first session in pair
            curr = session

            # feature_dict = extract_features(session)
            
            if prev is not None: # will never write to first/earliest cut file in sequence
                # compare(prev, curr)
                # match(prev, curr)
                # write_cut(curr)
                pass



"""

batch main fxn takes in study and does procedure across all pairs of sequenntial sessions

main fxn takes directory or session1 folder, session2 folder. 
    If directory assert only two sessions
    Figure out which session follows which
    Extract waveforms from cut
    Match waveforns to units from tetrode file (use sort by spike/cell fn)
    Return dictionary of spike waveforms for each unit (SpikeClusterBatch --> SpikeCluster --> Spike)

"""
def match_session_units(session_1 : Session, session_2: Session):
    """
    Input is two sequential Session() instances from study workspace

    session_2 follows session_1

    Returns 
    """

    assert isinstance(session_1, Session), 'Make sure inputs are of Session() class type'
    assert isinstance(session_2, Session), 'Make sure inputs are of Session() class type'

    ### TO DO

    # extracts features for every Spike in every SpikeCluster in SpikeClusterBatch (inputs are SpikeClusterBatch)
    # unit_features = get_all_unit_features(SpikeCluster) --> Sorted colleciton of Spike() objects belonging to one unit

    # match waveforms to units (inputs are SpikeCluster)
    # matched_units = match_units(unit_featuress)

    # best_matches = produce remapping (inputs are SpikeCluster)

    # apply remapping(best_matches)

    best_matches = {0:0, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:0, 8:0, 9:0, 10:0, 11:0}

    # print(np.unique(session_1.session_data.data['cell_ensemble'].get_label_ids()))
    # print(np.unique(session_2.session_data.data['cell_ensemble'].get_label_ids()))

    # print(np.unique(session_1.session_data.data['spike_cluster'].cluster_labels))
    # print(np.unique(session_2.session_data.data['spike_cluster'].cluster_labels))

    cut_file_path = session_2.session_metadata.file_paths['cut']

    cut_data, header_data = get_current_cut_data(cut_file_path)

    new_cut_data = apply_remapping(cut_data, best_matches)

    new_cut_file_path = format_new_cut_file_name(cut_file_path)

    write_cut(new_cut_file_path, new_cut_data, header_data) 

    return new_cut_file_path

