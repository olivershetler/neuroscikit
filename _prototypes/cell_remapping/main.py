import os, sys
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import time

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)


from _prototypes.cell_remapping.src.settings import settings_dict
from x_io.rw.axona.batch_read import make_study
from _prototypes.cell_remapping.src.remapping import compute_remapping

""" 

TO DO :

- make folder for every animal and cell, for every cell put ratemap for each sesesion comparison (e.g. ses1-ses2 + distance in title, then ses2-ses3 ratemaps side by side + distance) DOING
- make plots with activity bar next to ratemap DOING
- CHANGE SIZE OF OBJECT RATEMAP, make it 3 by 3 (9 squares) and make sure your code puts the middle square with all the density 
- DOUBLE CHECK what values you need to use when putting all density in the pobject location, 1 is incorrect, maybe use average Hz ratemap value? Or use np.max? Double check this

"""

def main():
    start_time = time.time()
    root = tk.Tk()
    root.withdraw()
    data_dir = filedialog.askdirectory(parent=root,title='Please select a data directory.')
    study = make_study(data_dir,settings_dict)
    study.make_animals()
    output = compute_remapping(study, settings_dict)

    if 'rate' in output:
        df = pd.DataFrame(output['rate'])
        df.to_csv(PROJECT_PATH + '/_prototypes/cell_remapping/output' + '/rate_remapping.csv')
    if 'object' in output:
        df = pd.DataFrame(output['object'])
        df.to_csv(PROJECT_PATH + '/_prototypes/cell_remapping/output' + '/obj_remapping.csv')

    print('Total run time: ' + str(time.time() - start_time))

if __name__ == '__main__':
    main()

