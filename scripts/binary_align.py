import os, sys 
import tkinter as tk
from tkinter import filedialog
import time
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from x_io.rw.axona.batch_read import make_study

# def _read_single_set(fileset):
#     filelist = list(fileset)

#     for file in filelist:
#         if 'cut' in file:
#             cut = file 
#         elif '.pos' in file:
#             pos = file 
#         else:
#             tet = file

#     event_times = read_


#     return event_times, x, y, position_time

def compute_binary_spike_trains_2D(event_times, x, y, position_time):
    binary_event_times = np.zeros((len(position_time)-1,len(event_times)))
    for i in range(len(event_times)):
        event_train = event_times[i]

        cts, _ = np.histogram(event_train, bins=position_time.squeeze())
        cts[cts > 0] = 1

        assert len(np.unique(cts)) == 2

        binary_event_times[:,i] = cts

    position_array = np.array(list(map(lambda i: [x[i], y[i]], np.arange(len(position_time))))).squeeze()

    return binary_event_times, position_array
        

if __name__ == '__main__':

    # USING NEUROSCIKIT V1
    start_time = time.time()
    root = tk.Tk()
    root.withdraw()
    data_dir = filedialog.askdirectory(parent=root,title='Please select a data directory.')

    animal = {'animal_id': '001'}
    devices = {'axona_led_tracker': True, 'implant': True}
    implant = {'implant_id': '001'}

    session_settings = {'animal': animal, 'devices': devices, 'implant': implant, 'channel_count': 4} 

    """ FOR YOU TO EDIT """
    settings = {'session':  session_settings, 'useMatchedCut': False, 'smoothing_factor': None} 
    """ FOR YOU TO EDIT """


    study = make_study(data_dir,settings_dict=settings)
    study.make_animals()

    # takes first animal (tetrode 1) and first session
    # change to for loop through animals then sessions for batch
    event_times = study.animals[0].sessions['session_1'].get_cell_data()['cell_ensemble'].get_event_times()[0]
    position_object = study.animals[0].sessions['session_1'].get_position_data()['position']
    binary_spike_trains, position_array = compute_binary_spike_trains_2D(event_times, position_object.x, position_object.y, position_object.t)
