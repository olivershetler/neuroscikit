import tkinter as tk
from tkinter import filedialog
import time
import os
import sys
import pickle
import pandas as pd
import openpyxl as xl
import numpy as np
import traceback

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from x_io.rw.axona.batch_read import make_study

pd.options.mode.chained_assignment = None


def adjust_neurofunc_to_matched_mapping(file_dir, data_dir, settings):
    # wb = xl.open_workbook(path, encoding_override='latin1')
    print(file_dir)
    xls = pd.read_excel(file_dir, sheet_name=None)
    # encoding=sys.getfilesystemencoding())
    # encoding='utf-8')
    no_matched_cut_df = xls.pop('Summary')

    # wb = xl.Workbook()
    # sum_sheet = wb['Sheet']
    # sum_sheet.title = 'Summary'
    # sum_sheet['A' + str(1)] = 'Session'
    # sum_sheet['B' + str(1)] = 'Tetrode'
    # sum_sheet['C' + str(1)] = 'Cell ID'

    headers = list(no_matched_cut_df.columns)
    headers.insert(3, 'Unmatched ID')

    adjusted_df = pd.DataFrame(columns=headers)
    matched_df = pd.DataFrame(columns=headers)
    unmatched_df = pd.DataFrame(columns=headers)

    subdirs = np.sort([ f.path for f in os.scandir(data_dir) if f.is_dir() ])
    count = 1
    for subdir in subdirs:
        study = make_study(subdir,settings_dict=settings)
        study.make_animals()
        count += 1

        for animal in study.animals:

            matched_lbls = np.unique(np.concatenate(list(map(lambda x: np.unique(animal.sessions[x].get_spike_data()['spike_cluster'].cluster_labels), list(animal.sessions.keys())))))
            print('matched')
            print(matched_lbls)
            if matched_lbls[0] == 0:
                empty_cell = sorted(set(range(matched_lbls[1], matched_lbls[-1] + 1)).difference(matched_lbls))
            else:
                empty_cell = sorted(set(range(matched_lbls[0], matched_lbls[-1] + 1)).difference(matched_lbls))

            if len(empty_cell) >= 1:
                empty_cell = empty_cell[0]
            else:
                empty_cell = matched_lbls[-1] + 1

            mapping_output_paths = np.sort([ f.path for f in os.scandir(subdir) if 'mappings' in f.path ])
            # print([f.path for f in os.scandir(data_dir) ])
            # print(mapping_output_path)
            assert len(mapping_output_paths) == 4
            tet_id = animal.animal_id.split('_tet')[-1]
            mapping_output_path = [x for x in mapping_output_paths if int(x.split('_mappings')[0][-1]) == int(tet_id)]
            assert len(mapping_output_path) == 1
            mapping_output_path = mapping_output_path[0]

            with open(mapping_output_path, 'rb') as handle:
                            mapping_output = pickle.load(handle)

            for ses in mapping_output:
                ses_mapping_output = mapping_output[ses]

                if 'first_ses_map_dict' in ses_mapping_output:
                    ses_key = 'session_' + str(ses)
                    session_tet_path = animal.sessions[ses_key].session_metadata.file_paths['tet']
                    adjusted_df, matched_df, unmatched_df = adjust_mapping(session_tet_path, ses_mapping_output['first_ses_map_dict'], adjusted_df, matched_df, unmatched_df, no_matched_cut_df, empty_cell)
                
                ses_key = 'session_' + str(ses+1)
                session_tet_path = animal.sessions[ses_key].session_metadata.file_paths['tet']
                adjusted_df, matched_df, unmatched_df = adjust_mapping(session_tet_path, ses_mapping_output['map_dict'], adjusted_df, matched_df, unmatched_df, no_matched_cut_df, empty_cell)
    
    save_path = file_dir.split('.xlsx')[0] + '_matched.xlsx'
    with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
        adjusted_df = adjusted_df.sort_values(['Session', 'Tetrode', 'Cell ID'])
        adjusted_df.to_excel(writer, sheet_name="Summary", index=True)

        matched_df = matched_df.sort_values(['Session', 'Tetrode', 'Cell ID'])
        matched_df.to_excel(writer, sheet_name="Matched", index=True)

        unmatched_df = unmatched_df.sort_values(['Session', 'Tetrode', 'Cell ID'])
        unmatched_df.to_excel(writer, sheet_name="Unmatched", index=True)
        
    print('saved at ' + save_path)

def adjust_mapping(session_tet_path, map_dict, adjusted_df, matched_df, unmatched_df, no_matched_cut_df, empty_cell):
    session_signature = str(session_tet_path.split("/")[-1][:-2])
    tetrode_id = session_tet_path.split("/")[-1][-1]
    for unmatched_cell_id in map_dict:
        matched_cell_id = map_dict[unmatched_cell_id]
        # print(type(no_matched_cut_df['Tetrode'][0]), type(no_matched_cut_df['Cell ID'][0]))
        print(session_signature, tetrode_id, unmatched_cell_id, map_dict)
        row = no_matched_cut_df.loc[(no_matched_cut_df['Session'] == session_signature) & (no_matched_cut_df['Tetrode'] == int(tetrode_id)) &
                                (no_matched_cut_df['Cell ID'] == int(unmatched_cell_id))]
        print(row, row.shape)
        assert len(row) == 1
        row['Cell ID'] = int(matched_cell_id)
        row['Unmatched ID'] = int(unmatched_cell_id)


        # if matched_cell_id >= empty cell # can get this bcs mapping_output has all sessions

        # adjusted_df.loc[len(adjusted_df.index)] = row
        adjusted_df = pd.concat([adjusted_df, row])
        # adjusted_df.append(row)
        print(matched_cell_id, empty_cell)
        if matched_cell_id < empty_cell:
            matched_df = pd.concat([matched_df, row])
        else:
            assert matched_cell_id > empty_cell
            unmatched_df = pd.concat([unmatched_df, row])
             

    return adjusted_df, matched_df, unmatched_df

# def adjust_mapping(mapping_output_path, map_dict, adjusted_df):
#     pass

if __name__ == '__main__':

    animal = {'animal_id': '001', 'species': 'mouse', 'sex': 'F', 'age': 1, 'weight': 1, 'genotype': 'type', 'animal_notes': 'notes'}
    devices = {'axona_led_tracker': True, 'implant': True}
    implant = {'implant_id': '001', 'implant_type': 'tetrode', 'implant_geometry': 'square', 'wire_length': 25, 'wire_length_units': 'um', 'implant_units': 'uV'}

    session_settings = {'channel_count': 4, 'animal': animal, 'devices': devices, 'implant': implant}

    """ FOR YOU TO EDIT """
    settings = {'ppm': 511, 'session':  session_settings, 'smoothing_factor': 3, 'useMatchedCut': True}
    """ FOR YOU TO EDIT """

    # possible saves are:
    # 1 csv per session (all tetrode and indiv): 'one_per_session' --> 5 sheets (summary of all 4 tetrodes, tet1, tet2, tet3, tet4)
    # 1 csv per animal per tetrode (all sessions): 'one_per_animal_tetrode' --> 4 sheets one per tet 
    # 1 csv per animal (all tetrodes & sessions): 'one_for_parent' --> 1 sheet
    """ FOR YOU TO EDIT """

    start_time = time.time()
    root = tk.Tk()
    root.withdraw()
    data_dir = filedialog.askdirectory(parent=root,title='Please select a data directory.')
    file_dir = filedialog.askopenfilename(parent=root,title='Please select the csv file.')
    

    adjust_neurofunc_to_matched_mapping(file_dir, data_dir, settings)