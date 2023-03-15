import os, sys
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import time
import numpy as np
import traceback

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)


from _prototypes.cell_modelling.src.settings import settings_dict as settings
from x_io.rw.axona.batch_read import make_study
from _prototypes.cell_modelling.src.models import get_models


def main():
    start_time = time.time()
    root = tk.Tk()
    root.withdraw()
    data_dir = filedialog.askdirectory(parent=root,title='Please select a data directory.')

        ########################################################################################################################

    """ OPTION 1 """
    """ RUNS EVERYTHING UNDER PARENT FOLDER (all subfolders loaded first) """
    # output_path = data_dir + '/model_output/'
    # study = make_study(data_dir,settings_dict=settings)
    # study.make_animals()
    # batch_map(study, settings, data_dir)

    """ OPTION 2 """
    """ RUNS EACH SUBFOLDER ONE AT A TIME """
    subdirs = np.sort([ f.path for f in os.scandir(data_dir) if f.is_dir() ])
    for subdir in subdirs:
        try:
            output_path = subdir + '/model_output/'
            study = make_study(subdir,settings_dict=settings)
            study.make_animals()

            # for seskey in study.animals[0].sessions:
            #     sesp = study.animals[0].sessions[seskey].session_metadata.file_paths['cut']
            #     print(sesp)
            # stop()
                

            get_models(study, settings, subdir)

        except Exception:
            print(traceback.format_exc())
            print('DID NOT WORK FOR DIRECTORY ' + str(subdir))





if __name__ == '__main__':
    main()
