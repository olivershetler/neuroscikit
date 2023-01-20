import os, sys
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import time

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from _prototypes.waveform_plotter.src.batch_plots import batch_plots
from _prototypes.waveform_plotter.src.settings import settings_dict
from x_io.rw.axona.batch_read import make_study

def main():
    start_time = time.time()
    root = tk.Tk()
    root.withdraw()
    data_dir = filedialog.askdirectory(parent=root,title='Please select a data directory.')


    study = make_study(data_dir, settings_dict)
    study.make_animals()

    batch_plots(study, settings_dict, data_dir)

    print('Total run time: ' + str(time.time() - start_time))

if __name__ == '__main__':
    main()

