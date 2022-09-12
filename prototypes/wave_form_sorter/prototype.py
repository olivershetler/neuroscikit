import os
import sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
print(PROJECT_PATH)


from core.data_spikes import (
    SpikeTrain,
    SpikeTrainBatch,
    Spike,
    SpikeCluster,
    SpikeClusterBatch,
)

from x_io.axona.read_tetrode_and_cut import (
    load_spike_train_from_paths,
    _read_cut,
    _format_spikes
)

from core.data_study import (
    Study,
    Event,
    Animal
)

from core.core_utils import (
    make_seconds_index_from_rate
)

from prototypes.wave_form_sorter.sort_waveforms_by_session import sort_waveforms_by_session
from prototypes.wave_form_sorter.match_waveforms_by_session import match_waveforms_by_session


if __name__ == '__main__':
    print('Running Prototype')

    prototype_dir = os.getcwd()
    print(prototype_dir)

    # parent = os.path.dirname(prototype_dir)
    parent_dir = os.path.dirname(prototype_dir)
    sys.path.append(parent_dir)
    print(parent_dir)

    # top_dir = os.path.dirname(parent_dir)
    # print(top_dir)

    data_dir = parent_dir + r'\neuroscikit_test_data\sequential_axona_sessions'
    print(data_dir)

    # Test data we are using has two sets of sequential sessions --> extract

    files = os.listdir(data_dir)

    test1_34 = []
    test1_35 = []
    test2_34 = []
    test2_35 = []

    for f in files:
        if 'Test1' in f and '34' in f:
            test1_34.append(f)
        elif 'Test1' in f and '35' in f:
            test1_35.append(f)
        elif 'Test2' in f and '34' in f:
            test2_34.append(f)
        elif 'Test2' in f and '35' in f:
            test2_35.append(f)

    # Get tet and cut files from inside folders

    # session1 = test1_34
    # session2 = test2_34

    session1 = test1_35
    session2 = test2_35

    assert len(session1) == len(session2)

    session1_tets = []
    session2_tets = []

    for i in range(len(session1)):
        if 'cut' in session1[i]:
            session1_cut = session1[i]
        if 'cut' in session2[i]:
            session2_cut = session2[i]
        file_session_1 = session1[i]
        file_session_2 = session2[i]
        out1 = file_session_1.split('.')[-1]
        out2 = file_session_2.split('.')[-1]
        if out1.isnumeric() and 'clu' not in file_session_1:
            session1_tets.append(session1[i])
        if out2.isnumeric() and 'clu' not in file_session_2:
            session2_tets.append(session2[i])

    session1_cut_path = os.path.join(data_dir, session1_cut)
    session1_tet_path = os.path.join(data_dir, session1_tets[0])

    session2_cut_path = os.path.join(data_dir, session2_cut)
    session2_tet_path = os.path.join(data_dir, session2_tets[0])

    # read data from cut and tet files

    with open(session1_cut_path, 'r') as cut_file1, open(session1_tet_path, 'rb') as tetrode_file1:
        cut_data1 = _read_cut(cut_file1)
        tetrode_data1 = _format_spikes(tetrode_file1)
        # ts, ch1, ch2, ch3, ch4, spikeparam

    with open(session2_cut_path, 'r') as cut_file2, open(session2_tet_path, 'rb') as tetrode_file2:
        cut_data2 = _read_cut(cut_file2)
        tetrode_data2 = _format_spikes(tetrode_file2)
        # ts, ch1, ch2, ch3, ch4, spikeparam

    # Make dictionaries for core classes

    sample_length1 =  tetrode_data1[-1]['duration']
    sample_rate1 = tetrode_data1[-1]['samples_per_spike']

    session_dict1 = {
        'spike_times': tetrode_data1[0].squeeze().tolist(),
        'cluster_labels': cut_data1,
        'ch1': tetrode_data1[1],
        'ch2': tetrode_data1[2],
        'ch3': tetrode_data1[3],
        'ch4': tetrode_data1[4],
    }

    sample_length2 =  tetrode_data2[-1]['duration']
    sample_rate2 = tetrode_data2[-1]['samples_per_spike']

    session_dict2 = {
        'spike_times': tetrode_data2[0].squeeze().tolist(),
        'cluster_labels': cut_data2,
        'ch1': tetrode_data2[1],
        'ch2': tetrode_data2[2],
        'ch3': tetrode_data2[3],
        'ch4': tetrode_data2[4],
    }

    assert sample_length1 == sample_length2
    assert sample_rate1 == sample_rate2

    study_dict = {
        'sample_length': sample_length1,
        'sample_rate': sample_rate1,
        'animal_ids': []
    }

    animal_dict = {
        'id': '0',
    }

    animal_dict[0] = session_dict1
    animal_dict[1] = session_dict2

    # Make study + add animal with sessions, can also add sessions one at a time after making animal instance

    study = Study(study_dict)

    study.add_animal(animal_dict)

    animal = study.animals[0]

    agg_waveform_dict = sort_waveforms_by_session(animal, study)
    matched = match_waveforms_by_session

    # still need to update cell references after matchings