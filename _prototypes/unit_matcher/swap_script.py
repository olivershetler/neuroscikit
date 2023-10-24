# Outside imports
import os, sys
import pandas as pd
import numpy as np
import traceback
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import time

# Set necessary paths / make project path = ...../neuroscikit/
pth = os.getcwd()
sys.path.append(pth)
os.chdir(pth)
print(pth)

# Internal imports

# Read write modules
from x_io.rw.axona.batch_read import make_study
from _prototypes.unit_matcher.read_axona import read_sequential_sessions, temp_read_cut

# Unit matching modules
from _prototypes.unit_matcher.session import compare_sessions
from _prototypes.unit_matcher.waveform import time_index, derivative, derivative2, morphological_points


def apply_remapping(cut_data, map_dict: dict):

    all_labels = np.unique(cut_data)
    for label in all_labels:
        if label not in map_dict:
            map_dict[label] = label
    new_cut_data = list(map(map_dict.get, cut_data))

    if len(map_dict)!= 0:
        # print('apply')
        # print(len(cut_data))
        # print(map_dict)
        new_counts = np.unique(new_cut_data, return_counts=True)[1]
        # print(new_counts)
        old_counts = np.unique(cut_data, return_counts=True)[1]
        # print(old_counts)
        # print('done')
    return new_cut_data

def get_current_cut_data(cut_session):
    with open(cut_session, 'r') as open_cut_file:
        cut_data, header_data =  temp_read_cut(open_cut_file)
    return cut_data, header_data

def format_cut(session, map_dict):
    cut_file_path = session.session_metadata.file_paths['cut']
    cut_data, header_data = get_current_cut_data(cut_file_path)
    new_cut_data = apply_remapping(cut_data, map_dict)
    # new_cut_file_path = format_new_cut_file_name(cut_file_path)
    new_cut_file_path = cut_file_path
    return new_cut_file_path, new_cut_data, header_data

def write_cut(cut_file, cut_data, header_data):
    """This function takes a cut file and updates the list of the neuron numbers in the file."""
    with open(cut_file, 'r') as open_cut_file:
        data = open_cut_file.read()

    extract_cut = False
    with open(cut_file, 'w') as open_cut_file:
        for line in header_data:
            open_cut_file.writelines(line)
            if 'Exact_cut' in line:  # finding the beginning of the cut values
                extract_cut = True
            if extract_cut: # start write to cut file
                open_cut_file.write(" ".join(map(str, cut_data)))
    open_cut_file.close()
    print('Wrote matched cut file ' + str(cut_file))

def batch_swap(study, settings, workdir, swap_df):
    for animal in study.animals:
        max_matched_cell_count = max(list(map(lambda x: max(animal.sessions[x].get_cell_data()['cell_ensemble'].get_label_ids()), animal.sessions)))
        map_dict = {}
        for cell_label in range(1,int(max_matched_cell_count)+1):
            for i in range(len(list(animal.sessions.keys()))):
                seskey = 'session_' + str(i+1)
                if seskey not in map_dict:
                    map_dict[seskey] = {}
                # print(seskey)
                ses = animal.sessions[seskey]
                path = ses.session_metadata.file_paths['tet']
                fname = str(path.split('/')[-1].split('.')[0])
                aid = fname.split('_')[0]
                date = fname.split('_')[1]
                date = date[:8]
                tet = animal.animal_id.split('tet')[1]

                ensemble = ses.get_cell_data()['cell_ensemble']

                swap_df_row_id = aid + '_' + date + '_' + tet + '_' + str(cell_label)

                # print(swap_df_row_id)
                if cell_label in ensemble.get_cell_label_dict() and swap_df_row_id in list(swap_df['CATEGORY'].to_numpy()):
                    swap_df_row_ids = np.where(swap_df['CATEGORY'].to_numpy() == swap_df_row_id)[0]
                    for swap_df_row_id in swap_df_row_ids:
                        df_sample = swap_df.iloc[swap_df_row_id:swap_df_row_id+1]
                        row = df_sample['Instructions']
                        row = row.values[0]
                        # print(row)
                        if row != row:
                            pass 
                        else:
                            # format: "X to Y for sesZ"
                            # print(row)
                            row = str(row)
                            origin = row.split(' to ')[0]
                            target = row.split(' to ')[1].split(' for ')[0]
                            # ses_id_to_move = row.split(' to ')[1].split(' for ')[1]
                            # ses_id_to_move = ses_id_to_move.split('ses')[1]
                            ses_id_to_move = row.split('ses')[1]

                            map_dict['session_' + str(ses_id_to_move)][int(origin)] = int(target)
                            # print('print')
                            # print(map_dict)
        for i in range(len(list(animal.sessions.keys()))):
            seskey = 'session_' + str(i+1)
            if len(map_dict[seskey]) != 0:
                ses = animal.sessions[seskey]
                matched_cut_file_path, new_data, header_data = format_cut(ses, map_dict[seskey])
                write_cut(matched_cut_file_path, new_data, header_data)






if __name__ == '__main__':

    ######################################################## EDIT BELOW HERE ########################################################

    # MAKE SURE "field_sizes" IS THE LAST ELEMENT IN "csv_header_keys"
    csv_header = {}
    csv_header_keys = ['name','date','depth','stim','information', 'shuffled_information_mean','shuffled_information_std','p_value_information',
                       'selectivity', 'shuffled_selectivity_mean', 'shuffled_selectivity_std', 'p_value_selectivity',
                       'sparsity', 'shuffled_sparsity_mean', 'shuffled_sparsity_std', 'p_value_sparsity',
                       'coherence', 'shuffled_coherence_mean', 'shuffled_coherence_std', 'p_value_coherence',
                       'shuffled_offset']
    for key in csv_header_keys:
        csv_header[key] = True

    tasks = {}
    task_keys = ['information']
    for key in task_keys:
        tasks[key] = True

    plotTasks = {}
    plot_task_keys = ['Spikes_Over_Position_Map', 'Tuning_Curve_Plots', 'Firing_Rate_vs_Speed_Plots', 'Firing_Rate_vs_Time_Plots','autocorr_map', 'binary_map','rate_map', 'occupancy_map']
    for key in plot_task_keys:
        plotTasks[key] = True

    animal = {'animal_id': '001', 'species': 'mouse', 'sex': 'F', 'age': 1, 'weight': 1, 'genotype': 'type', 'animal_notes': 'notes'}
    devices = {'axona_led_tracker': True, 'implant': True}
    implant = {'implant_id': '001', 'implant_type': 'tetrode', 'implant_geometry': 'square', 'wire_length': 25, 'wire_length_units': 'um', 'implant_units': 'uV'}

    session_settings = {'channel_count': 4, 'animal': animal, 'devices': devices, 'implant': implant}

    """ FOR YOU TO EDIT """
    settings = {'ppm': 485, 'session':  session_settings, 'smoothing_factor': 3, 'useMatchedCut': True}
    """ FOR YOU TO EDIT """

    tasks['disk_arena'] = True # -->
    settings['tasks'] = tasks # --> change tasks array to change tasks are run
    settings['plotTasks'] = plotTasks # --> change plot tasks array to change asks taht are plotted
    settings['header'] = csv_header # --> change csv_header header to change tasks that are saved to csv

    """ FOR YOU TO EDIT """
    settings['naming_type'] = 'LEC'
    settings['speed_lowerbound'] = 0
    settings['speed_upperbound'] = 99
    settings['end_cell'] = None
    settings['start_cell'] = None
    settings['saveData'] = True
    settings['ratemap_dims'] = (32, 32)
    settings['saveMethod'] = 'one_for_parent'
    # possible saves are:
    # 1 csv per session (all tetrode and indiv): 'one_per_session' --> 5 sheets (summary of all 4 tetrodes, tet1, tet2, tet3, tet4)
    # 1 csv per animal per tetrode (all sessions): 'one_per_animal_tetrode' --> 4 sheets one per tet 
    # 1 csv per animal (all tetrodes & sessions): 'one_for_parent' --> 1 sheet
    """ FOR YOU TO EDIT """


    start_time = time.time()
    root = tk.Tk()
    root.withdraw()
    data_dir = filedialog.askdirectory(parent=root,title='Please select a data directory.')

    ########################################################################################################################

    """ OPTION 1 """
    """ RUNS EVERYTHING UNDER PARENT FOLDER (all subfolders loaded first) """
    # study = make_study(data_dir,settings_dict=settings)
    # study.make_animals()
    # batch_swap(study, settings, data_dir)

    # """ OPTION 2 """
    # """ RUNS EACH SUBFOLDER ONE AT A TIME """
    # subdirs = np.sort([ f.path for f in os.scandir(data_dir) if f.is_dir() ])
    # for subdir in subdirs:
    #     try:
    #         study = make_study(subdir,settings_dict=settings)
    #         study.make_animals()
    #         batch_swap(study, settings, subdir)
    #     except Exception:
    #         print(traceback.format_exc())
    #         print('DID NOT WORK FOR DIRECTORY ' + str(subdir))

    """ OPTION 3 """
    """ RUNS EACH SUBFOLDER ONE AT A TIME """
    subdirs = np.sort([ f.path for f in os.scandir(data_dir) if f.is_dir() ])
    swap_df_path = r"C:\Users\aaoun\OneDrive - cumc.columbia.edu\Desktop\HussainiLab\neuroscikit_test_data\Sifting_Through_Cells.xlsx"
    swap_df = pd.read_excel(swap_df_path, sheet_name='ANT')
    swap_df['CATEGORY'] = swap_df['CATEGORY'].astype(str)
    swap_df['CATEGORY'] = swap_df['CATEGORY'].str.replace('.png', '')
    for subdir in subdirs:
        subdir = str(subdir)
        subdirs2 = np.sort([ f.path for f in os.scandir(subdir) if f.is_dir() ])
        for subdir2 in subdirs2:
            study = make_study(subdir2,settings_dict=settings)
            study.make_animals()
            batch_swap(study, settings, subdir2, swap_df)
