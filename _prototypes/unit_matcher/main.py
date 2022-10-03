import os, sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.study_space import Session
from x_io.rw.axona.batch_read import _read_cut

"""

This module reads axona cut and tetrode files.
It then extracts the spike waveforms from the cut file.
It then matches the spike waveforms to the units in the tetrode file.
It then produces a dictionary of the spike waveforms for each unit.
It then extracts features from the spike waveforms for each unit.
It then matches the spike waveforms to the units in the tetrode file.
It then produces a remapping of the units in the tetrode file.
It then applies the remapping to the cut file data.
It then writes the remapped cut file data to a new cut file.
Read, Retain, Map, Write
"""





"""

batch main fxn takes in study and does procedure across all pairs of sequenntial sessions

main fxn takes directory or session1 folder, session2 folder. 
    If directory assert only two sessions
    Figure out which session follows which
    Extract waveforms from cut
    Match waveforns to units from tetrode file (use sort by spike/cell fn)
    Return dictionary of spike waveforms for each unit (SpikeClusterBatch --> SpikeCluster --> Spike)

"""

def _apply_remapping(cut_session, map_dict: dict):
    """
    Input is a dictionary mapping change in label from session 1 to session 2
    If label is None, unit has no match in other sessions

    Session 2 unit labels will be changed to match session 1 unit labels

    e.g. {5:4, 6:2, 7:1}

    If unit in session 1 is unmatched in session 2, value of key will be None e.g. {2: None, 1:None} and no change will be made to units in session 2

    If unit in session 2 is unmatchd in session 1, key of value will be 'None' {'None': 2, 'None': 5}

    Only ever write to cut file 2


    """
    with open(cut_session, 'r') as open_cut_file:
        cut_data, header_data =  _temp_read_cut(open_cut_file)

        new_cut_data = list(map(map_dict.get, cut_data))

    return new_cut_data, header_data


def _temp_read_cut(cut_file):
    """This function takes a pre-opened cut file and updates the list of the neuron numbers in the file."""
    cut_values = None
    extract_cut = False
    header_data = []
    for line in cut_file:
        if not extract_cut:
            header_data.append(line)
        if 'Exact_cut' in line:  # finding the beginning of the cut values
            extract_cut = True
        if extract_cut:  # read all the cut values
            cut_values = str(cut_file.readlines())
            for string_val in ['\\n', ',', "'", '[', ']']:  # removing non base10 integer values
                cut_values = cut_values.replace(string_val, '')
            cut_values = [int(val) for val in cut_values.split()]
        else:
            continue
    if cut_values is None:
        raise ValueError('Either file does not exist or there are no cut values found in cut file.')
        # cut_values = np.asarray(cut_values)
    return cut_values, header_data

def write_cut(cut_file, cut_data, header_data):
    """This function takes a cut file and updates the list of the neuron numbers in the file."""
    extract_cut = False
    with open(cut_file, 'w+') as open_cut_file:
        for line in header_data:
            open_cut_file.writelines(line)
            if 'Exact_cut' in line:  # finding the beginning of the cut values
                extract_cut = True
            if extract_cut: # start write to cut file
                open_cut_file.write(" ".join(map(str, cut_data)))
    open_cut_file.close()

def _format_new_cut_file_name(old_path):
    fp_split = old_path.split('.cut')
    assert len(fp_split) == 2
    new_fp = fp_split[0] + r'_matched.cut' 
    assert new_fp != old_path

    return new_fp

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

    # match waveforms to units (inputs are SpikeCluster)

    # best_matches = produce remapping (inputs are SpikeCluster)

    # apply remapping

    best_matches = {0:0, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:0, 8:0, 9:0, 10:0, 11:0}

    print(np.unique(session_1.session_data.data['cell_ensemble'].get_label_ids()))
    print(np.unique(session_2.session_data.data['cell_ensemble'].get_label_ids()))

    print(np.unique(session_1.session_data.data['spike_cluster'].cluster_labels))
    print(np.unique(session_2.session_data.data['spike_cluster'].cluster_labels))

    cut_file_path = session_2.session_metadata.file_paths['cut']

    new_cut_data, header_data = _apply_remapping(cut_file_path, best_matches)

    new_cut_file_path = _format_new_cut_file_name(cut_file_path)

    write_cut(new_cut_file_path, new_cut_data, header_data)

    return new_cut_file_path
