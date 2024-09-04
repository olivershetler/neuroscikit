import os, sys
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import time
import numpy as np
import traceback
import statsmodels.api as sm

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)


from _prototypes.cell_remapping.src.settings import settings_dict as settings
from _prototypes.cell_remapping.src.stats import get_max_matched_cell_count
from _prototypes.cell_remapping.src.utils import read_data_from_fname, check_cylinder, check_object_location
from _prototypes.cell_remapping.src.masks import make_object_ratemap, apply_disk_mask
from _prototypes.cell_remapping.src.processing import get_rate_map
import matplotlib.gridspec as gridspec

from x_io.rw.axona.batch_read import make_study
from library.study_space import Session, Study, SpatialSpikeTrain2D
# sklearn linear_model
from sklearn import linear_model
import matplotlib.pyplot as plt
# r2 score
from sklearn.metrics import r2_score

def split_endog_exog(y, X, p=0.7):
    """
    p is train-test split percentage
    """
    idx = int(p * len(y))

    trainY = y[:idx]
    testY = y[idx:]
    # print(X.shape, y.shape, idx)
    trainX = X[:,:idx]
    testX = X[:,idx:]

    # print(trainX.shape, trainY.shape, testX.shape, testY.shape)

    return trainX, trainY, testX, testY


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

    ses1_cts = []
    ses2_cts = []
    ses3_cts = []

    ses_cts = [ses1_cts, ses2_cts, ses3_cts]

    ses1_wins = []
    ses2_wins = []
    ses3_wins = []

    ses_wins = [ses1_wins, ses2_wins, ses3_wins]

    ses1_ids = []
    ses2_ids = []
    ses3_ids = []

    ses_ids = [ses1_ids, ses2_ids, ses3_ids]

    ses1_ctypes = []
    ses2_ctypes = []
    ses3_ctypes = []

    ses_ctypes = [ses1_ctypes, ses2_ctypes, ses3_ctypes]

    ses1_ramp_dirs = []
    ses2_ramp_dirs = []
    ses3_ramp_dirs = []

    ses_ramp_dirs = [ses1_ramp_dirs, ses2_ramp_dirs, ses3_ramp_dirs]

    """ OPTION 2 """
    """ RUNS EACH SUBFOLDER ONE AT A TIME """
    subdirs = np.sort([ f.path for f in os.scandir(data_dir) if f.is_dir() ])
    for subdir in subdirs:
        try:
            study = make_study(subdir,settings_dict=settings)
            study.make_animals()

            ses_wins, ses_cts, ses_ids, ses_ctypes = run_temporal_map(study, settings, ses_wins, ses_cts, ses_ids, ses_ctypes, ses_ramp_dirs)

        except Exception:
            print(traceback.format_exc())
            print('DID NOT WORK FOR DIRECTORY ' + str(subdir))

    # save ses_wins and ses_cts
    ses_wins_path = data_dir + '/ses_wins.npy'
    ses_cts_path = data_dir + '/ses_cts.npy'
    ses_ids_path = data_dir + '/ses_ids.npy'
    ses_ctypes_path = data_dir + '/ses_ctypes.npy'
    ses_ramp_dirs_path = data_dir + '/ses_ramp_dirs.npy'
    np.save(ses_wins_path, ses_wins)
    np.save(ses_cts_path, ses_cts)
    np.save(ses_ids_path, ses_ids)
    np.save(ses_ctypes_path, ses_ctypes)
    np.save(ses_ramp_dirs_path, ses_ramp_dirs)

def run_temporal_map(study, settings, ses_wins, ses_cts, ses_ids, ses_ctypes, ses_ramp_dirs):

    ses1_cts = ses_cts[0]
    ses2_cts = ses_cts[1]
    ses3_cts = ses_cts[2]

    ses1_wins = ses_wins[0]
    ses2_wins = ses_wins[1]
    ses3_wins = ses_wins[2]

    ses1_ids = ses_ids[0]
    ses2_ids = ses_ids[1]
    ses3_ids = ses_ids[2]

    ses1_ctypes = ses_ctypes[0]
    ses2_ctypes = ses_ctypes[1]
    ses3_ctypes = ses_ctypes[2]

    ses1_ramp_dirs = ses_ramp_dirs[0]
    ses2_ramp_dirs = ses_ramp_dirs[1]
    ses3_ramp_dirs = ses_ramp_dirs[2]

    output_path = r"C:\Users\aaoun\OneDrive - cumc.columbia.edu\Desktop\HussainiLab\neuroscikit_test_data\testing_temporal"

    ctype_csv = r"C:\Users\aaoun\OneDrive - cumc.columbia.edu\Desktop\HussainiLab\neuroscikit\df_full_LEC_assigned_keep_swapped_fixed.xlsx"
    ctype_df = pd.read_excel(ctype_csv)

    remapping_csv = r'C:\Users\aaoun\OneDrive - cumc.columbia.edu\Desktop\HussainiLab\neuroscikit\df_full_LEC.xlsx'
    remapping_df = pd.read_excel(remapping_csv)

    remapping_df['score'] = remapping_df['score'].astype(str)
    remapping_df = remapping_df[remapping_df['score'] == 'whole']

    for animal in study.animals:
        tetrode = animal.animal_id.split('_tet')[1]

        max_matched_cell_count = get_max_matched_cell_count(animal)

        for k in range(int(max_matched_cell_count)):
            cell_label = k + 1
            cell_dict = {}

            ctype = None 

            for i in range(len(list(['session_1', 'session_2', 'session_3']))):
                seskey = 'session_' + str(i+1)
                # ctype = None 

                ses = animal.sessions[seskey]
                ensemble = ses.get_cell_data()['cell_ensemble']
                pos_obj = ses.get_position_data()['position']
                ses = animal.sessions[seskey]
                path = ses.session_metadata.file_paths['tet']
                fname = path.split('/')[-1].split('.')[0]


                cylinder = check_cylinder(fname, settings['disk_arena'])
        
                stim, depth, name, date = read_data_from_fname(fname, settings['naming_type'], settings['type'])

                aid = name

                object_location = check_object_location(stim, True)

                ensemble = ses.get_cell_data()['cell_ensemble']
                
                if cell_label in ensemble.get_cell_label_dict():

                    cell = ensemble.get_cell_by_id(cell_label)

                    if seskey not in cell_dict:
                        cell_dict[seskey] = {}

                    remapping_df['name'] = remapping_df['name'].astype(str)
                    remapping_df['date'] = remapping_df['date'].astype(str)
                    remapping_df['tetrode'] = remapping_df['tetrode'].astype(int)
                    remapping_df['unit_id'] = remapping_df['unit_id'].astype(int)
                    remapping_df['depth'] = remapping_df['depth'].astype(int)
                    remapping_df['session_id'] = remapping_df['session_id'].astype(str)

                    remapping_row = remapping_df[(remapping_df['name'] == str(aid)) & (remapping_df['date'] == str(date)) & 
                                                           (remapping_df['tetrode'] == int(tetrode)) & 
                                                           (remapping_df['unit_id'] == int(cell_label)) &
                                                           (remapping_df['depth'] == int(depth)) &
                                                              (remapping_df['session_id'] == str(seskey))]
                    
                    # assert that there is only one row per cell
                    print(aid, date, tetrode, cell_label, depth, seskey)
                    print(remapping_df['name'].dtype, remapping_df['date'].dtype, remapping_df['tetrode'].dtype, remapping_df['unit_id'].dtype, remapping_df['depth'].dtype, remapping_df['session_id'].dtype)
                    print(remapping_row)
                    print(object_location, remapping_row['object_location'])

                    assert len(remapping_row) == 1
                    remapping_row = remapping_row.iloc[0]

                    assert str(object_location) == str(remapping_row['object_location'])

                    ctype_df['name'] = ctype_df['name'].astype(str)
                    ctype_df['date'] = ctype_df['date'].astype(str)
                    ctype_df['tetrode'] = ctype_df['tetrode'].astype(int)
                    ctype_df['unit_id'] = ctype_df['unit_id'].astype(int)
                    ctype_df['depth'] = ctype_df['depth'].astype(int)
                    ctype_df['session_id'] = ctype_df['session_id'].astype(str)

                    ctype_row = ctype_df[(ctype_df['name'] == str(aid)) & (ctype_df['date'] == str(date)) &
                                                              (ctype_df['tetrode'] == int(tetrode)) &
                                                                (ctype_df['unit_id'] == int(cell_label)) &
                                                                (ctype_df['depth'] == int(depth)) &
                                                                (ctype_df['session_id'] == str(seskey))]
                    
                    # print(aid, date, tetrode, cell_label, depth, seskey)
                    # print(ctype_df['name'].dtype, ctype_df['date'].dtype, ctype_df['tetrode'].dtype, ctype_df['unit_id'].dtype, ctype_df['depth'].dtype, ctype_df['session_id'].dtype)
                    # print(ctype_row)
                    # print('NEXT')
                    # # print rows 861 to 864
                    # print(ctype_df.iloc[861:865])
                    # print(object_location, ctype_row['object_location'])

                    # print(len(ctype_row))

                    if len(ctype_row) >= 1:
                        for i in range(len(ctype_row)):
                            print('in here')
                            ctype_row_loc = ctype_row.iloc[i]
                            print(ctype_row_loc)
                            if i == 0:
                                ctype = ctype_row_loc['cell_type']
                            else:
                                if ctype == 'unassigned' and ctype_row_loc['cell_type'] != 'unassigned':
                                    ctype = ctype_row_loc['cell_type']
                            print(ctype)
                            print(ctype_row['cell_type'])
                            # stop()                        

                    # if str(tetrode) == '1' and str(cell_label) == '1':
                    #     stop()

                    spatial_spike_train = ses.make_class(SpatialSpikeTrain2D, 
                                    {   'cell': cell, 'position': pos_obj, 'speed_bounds': (settings['speed_lowerbound'], settings['speed_upperbound'])})   

    
                    rate_map_obj = spatial_spike_train.get_map('rate')

                    rate_map = get_rate_map(rate_map_obj, settings['ratemap_dims'], settings['normalizeRate'])

                    curr, curr_ratemap, disk_ids = apply_disk_mask(rate_map, settings, cylinder)
                    
                    cell_dict[seskey]['rate_map'] = curr_ratemap
                    cell_dict[seskey]['angle'] = object_location

                    T = np.max(cell.event_times)
                    # T = 15 * 60
                    dt = T/600

                    cts, _ = event_times_to_count(cell.event_times,T+dt,dt)

                    max_rate = len(cell.event_times) / T
                    # print(max_rate, T, len(cell.event_times))
                    max_rate = round(max_rate, 3)
                    cell_dict[seskey]['max_rate'] = max_rate

                    signal = cell.signal

                    cell_dict[seskey]['signal'] = signal

                    time_ramps = [] 
                    scaled_time_ramps = []
                    norm_time_ramps = []
                    dfbetas_agg = []
                    r2_scores = []
                    rsquared_scores = []
                    residuals_agg = []
                    betas_agg = []

                    # ramp_windows = np.arange(2, len(cts)+len(cts)/256, len(cts)/256)
                    ramp_windows = np.arange(2, len(cts)+1, 1)
                    # ramp_windows = np.arange(0,128,1)
                    twindow_ct = 1
                    max_t_score = 0
                    for twindow in ramp_windows:
                        ramp_values, scaled_ramp_values, norm_ramp_values = generate_time_ramps(len(cts), twindow)
                        # ramp_values, scaled_ramp_values, norm_ramp_values = generate_cosine_basis(len(cts), twindow, twindow_ct)
                        # ramp_values, scaled_ramp_values, norm_ramp_values = generate_cosine_basis(len(cts), T, twindow_ct)
                        
                        twindow_ct += 1
                        # twindow_ct += 0.1
                        # plt.plot(ramp_values)
                        # plt.show()
                    
                
                        time_ramps.append(ramp_values)
                        scaled_time_ramps.append(scaled_ramp_values)
                        norm_time_ramps.append(norm_ramp_values)

                        X = ramp_values
                        y = cts

                        X = X.reshape(1,-1)

                        print(X.shape, y.shape)

                        # trainX, trainY, testX, testY = split_endog_exog(y, X, p=0.99)
                        # trainX = trainX.T
                        # testX = testX.T
                        trainX = X 
                        testX = X
                        trainY = y
                        testY = y

                        # # fit linear model from sklearn
                        # clf = linear_model.PoissonRegressor(fit_intercept=True)
                        # clf.fit(trainX, trainY)
                        # # r2 score
                        # test_score = clf.score(testX, testY)
                        # # predict
                        # train_pred = clf.predict(trainX)

                        # fit linear model from statsmodels
                        # add constant to X
                        # trainX = sm.add_constant(trainX).T
                        # testX = sm.add_constant(testX).T

                        constant = np.ones(len(trainY))
                        # * np.mean(trainY) 
                        trainX = np.vstack((constant, trainX.flatten()))
                        constant = np.ones(len(testY)) 
                        # * np.mean(trainY) 
                        testX = np.vstack((constant, testX.flatten()))

                        print(trainX.shape, trainY.shape, testX.shape, testY.shape)
                        # fit model
                        model = sm.GLM(trainY, trainX.T, family=sm.families.Poisson(link=sm.families.links.log()))
                        # model = sm.GLM(trainY, trainX, family=sm.families.Poisson())
                        model_res = model.fit()
                        # predict
                        train_pred = model_res.predict(trainX.T)
                        test_pred = model_res.predict(testX.T)
                        # r2 score
                        test_score = r2_score(testY, test_pred)
                        train_score = r2_score(trainY, train_pred)
                        # beta coefficients
                        beta = model_res.params
                        beta = beta[1:]
                        # rsquared 
                        deviance = model_res.deviance
                        null_deviance = model_res.null_deviance
                        rsquared = (null_deviance - deviance)/null_deviance
                        # dfbetas
                        dfbetas = model_res.get_influence().dfbetas
                        dfbetas = dfbetas[:,-1]
                        # absolute value of dfbetas
                        dfbetas = np.abs(dfbetas)                    
                        # dfbetas = dfbetas / np.std(dfbetas)
                        # residuals
                        residuals = model_res.resid_response
                        residuals = np.abs(residuals)
                        # residuals = residuals / np.std(residuals)


                        dfbetas_agg.append(dfbetas)
                        r2_scores.append(train_score)
                        rsquared_scores.append(rsquared)
                        residuals_agg.append(residuals)
                        betas_agg.append(beta)

                        if train_score > max_t_score:
                            max_t_score = train_score
                            max_rsq = rsquared
                            max_twindow = twindow
                            max_train_pred = train_pred
                            max_trainY = trainY

                            cell_dict[seskey]['max_t_score'] = round(max_t_score, 3)
                            max_bin_ct = len(cts)
                            per = max_twindow / max_bin_ct
                            max_twindow_insecs = per * T / 60
                            max_twindow_insecs = round(max_twindow_insecs, 1)
                            cell_dict[seskey]['max_twindow'] = max_twindow_insecs
                            cell_dict[seskey]['max_train_pred'] = max_train_pred
                            cell_dict[seskey]['max_trainY'] = max_trainY

            if ctype is None:
                ctype = 'dropped'

            fig = plt.figure(figsize=(12,6))
            gs_main = gridspec.GridSpec(2, len(cell_dict), height_ratios=[4,18])

            for i, seskey in enumerate(cell_dict.keys()):
                max_t_score = cell_dict[seskey]['max_t_score']
                max_twindow = cell_dict[seskey]['max_twindow']
                max_train_pred = cell_dict[seskey]['max_train_pred']
                max_trainY = cell_dict[seskey]['max_trainY']
                max_rate = cell_dict[seskey]['max_rate']
                max_rate = round(max_rate, 1)
                max_t_score = round(max_t_score, 3)
                ax_top = fig.add_subplot(gs_main[0,i])
                ax_top2 = ax_top.twinx()

                tarray = np.arange(0, len(max_train_pred), 1) * dt
                ax_top2.plot(tarray, max_train_pred, color='red')
                ax_top.plot(tarray, max_trainY, color='grey')
                ax_top.set_xlabel('Time (s)')
                ax_top.set_ylabel('Fr (Hz)')
                max_twindow = float(max_twindow) * 60
                max_twindow = round(max_twindow, 1)
                ax_top.set_title('r2: ' + str(max_t_score) + ', Fr: ' + str(max_rate) + 'Hz, Window: ' + str(max_twindow) + 's')
                # plt.show()  

                # check if the first bin is close to 0 to determine if it is a ramp up or ramp down
                difference_to_max = np.abs(max_train_pred[0] - np.max(max_train_pred))
                difference_to_min = np.abs(max_train_pred[0] - np.min(max_train_pred))
                if difference_to_max < difference_to_min:
                    ramp_dir = 1
                else:
                    ramp_dir = 0

                print('ramp_dir' + str(ramp_dir))

                if seskey == 'session_1':
                    lbl_id = aid + '_' + date + '_' + depth + '_' + tetrode + '_' + str(cell_label) + '_' + str(object_location)
                    ses1_ids.append(lbl_id)
                    ses1_ctypes.append(ctype)
                    ses1_cts.append(max_t_score)
                    ses1_wins.append(max_twindow)
                    ses1_ramp_dirs.append(ramp_dir)

                elif seskey == 'session_2':
                    lbl_id = aid + '_' + date + '_' + depth + '_' + tetrode + '_' + str(cell_label) + '_' + str(object_location)
                    ses2_ids.append(lbl_id)
                    ses2_ctypes.append(ctype)
                    ses2_cts.append(max_t_score)
                    ses2_wins.append(max_twindow)
                    ses2_ramp_dirs.append(ramp_dir)

                elif seskey == 'session_3':
                    lbl_id = aid + '_' + date + '_' + depth + '_' + tetrode + '_' + str(cell_label) + '_' + str(object_location)
                    ses3_ids.append(lbl_id)
                    ses3_ctypes.append(ctype)
                    ses3_cts.append(max_t_score)
                    ses3_wins.append(max_twindow)
                    ses3_ramp_dirs.append(ramp_dir)



                            
            for i, seskey in enumerate(cell_dict.keys()):
                angle = cell_dict[seskey]['angle']
                ratemap = cell_dict[seskey]['rate_map']
                max_rate = cell_dict[seskey]['max_rate']
                gs_sub = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_main[1,i], height_ratios=[3,1])

                ax = fig.add_subplot(gs_sub[0,0])
                img = ax.imshow(ratemap, cmap='jet', aspect='equal')
                rate_title = str(max_rate) + ' Hz' 

                if angle != 'NO':
                    angle = int(angle)

                    _, obj_loc = make_object_ratemap(angle, new_size=settings['ratemap_dims'][0])
                                    
                    if angle == 0:
                        obj_loc['x'] += .5
                        obj_loc['y'] += 2
                    elif angle == 90:
                        obj_loc['y'] += .5
                        obj_loc['x'] -= 2
                    elif angle == 180:
                        obj_loc['x'] -= .5
                        obj_loc['y'] -= 2
                    elif angle == 270:
                        obj_loc['y'] -= .5
                        obj_loc['x'] += 2
                    ax.plot(obj_loc['x'], obj_loc['y'], 'k', marker='o', markersize=20)

                fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

                ax2 = fig.add_subplot(gs_sub[1,0])
                waveforms = cell_dict[seskey]['signal']
                for ch_id in range(4):
                    if ch_id != len(waveforms):
                        ch = waveforms[:,ch_id,:]
                        idx = np.random.choice(len(ch), size=200)
                        waves = ch[idx, :]
                        avg_wave = np.mean(ch, axis=0)

                        ax2.plot(np.arange(int(50*ch_id+5*ch_id),int(50*ch_id+5*ch_id+50),1), ch[idx,:].T, c='grey')
                        ax2.plot(np.arange(int(50*ch_id+5*ch_id),int(50*ch_id+5*ch_id+50),1), avg_wave, c='k', lw=2)
                        # ax2.set_xlim([-25,200])
                        ax2.set_xlim([-5,200])
                        ax2.spines['top'].set_visible(False)
                        ax2.spines['bottom'].set_visible(False)
                        ax2.spines['right'].set_visible(False)
                        ax2.spines['left'].set_visible(False)

                        ax2.tick_params(
                            axis='x',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False)
                        

                        ax2.tick_params(
                            axis='y',          # changes apply to the x-axis
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False)

                        ax2.set_xticks([])
                        ax2.set_yticks([])
                            
            title = str(ctype) + ' cell - animal: ' + str(aid) + ', date: ' + str(date) + ', depth: ' + str(depth) + ', tetrode: ' + str(tetrode) + ', unit: ' + str(cell_label)
            # fig.suptitle(str(ctype) + ' cell ' + str(animal), fontweight='bold')
            fig.suptitle(title)
            fig.tight_layout()
            save_path = output_path + '/' + str(aid) + '_' + str(date) + '_' + str(tetrode) + '_' + str(cell_label) + '.png'
            fig.savefig(save_path , dpi=360)
            plt.close()
                            
                    # # plot 
                    # fig = plt.figure(figsize=(8,3))
                    # ax = plt.subplot(1,1,1)
                    # ax2 = ax.twinx()
                    # ax2.plot(max_train_pred, color='red')
                    # ax.plot(max_trainY, color='blue')
                    # ax.set_title('R2: ' + str(max_t_score) + ' Fr: ' + str(max_rate) + ' Window: ' + str(max_twindow))
                    # plt.show()

                    # time_ramps = np.asarray(time_ramps)
                    # scaled_time_ramps = np.asarray(scaled_time_ramps)
                    # norm_time_ramps = np.asarray(norm_time_ramps)
                    # dfbetas_agg = np.asarray(dfbetas_agg)
                    # r2_scores = np.asarray(r2_scores)
                    # rsquared_scores = np.asarray(rsquared_scores)
                    # residuals_agg = np.asarray(residuals_agg)
                    # betas_agg = np.asarray(betas_agg)

                    # # plot imshow of X
                    # fig = plt.figure(figsize=(8,3))
                    # plt.imshow(scaled_time_ramps, aspect='auto', cmap='jet')
                    # plt.show()
                    # # plot r2 scores
                    # fig = plt.figure(figsize=(8,3))
                    # ax = plt.subplot(1,1,1)
                    # ax.plot(ramp_windows, r2_scores, color='k')

                    # idx = np.where(r2_scores < 0.1)[0]
                    # ramp_windows_copy = ramp_windows.copy()
                    # ramp_windows_copy = np.array(ramp_windows_copy, dtype=float)
                    # ramp_windows_copy[idx] = np.nan
                    # r2_scores_copy = r2_scores.copy()
                    # r2_scores_copy[idx] = np.nan
                    # ax.plot(ramp_windows_copy, r2_scores_copy, color='red')
                    # axt = ax.twinx()
                    # axt.plot(ramp_windows, betas_agg, color='blue')
                    # plt.show()

                    # # #cumulative fit with all ramps
                    # # X_cumulative = time_ramps
                    # # # X_cumulative = X_cumulative.reshape(1,-1)
                    # # print(X_cumulative.shape, cts.shape)
                    # # constant = np.ones(len(cts)).reshape(1,-1)
                    # # # constant is of shape (1, len(cts)) and X_cumulative is of shape (20, len(cts))
                    # # # need shape to be (len(cts), 21) so that each column is a ramp and first column is constant
                    # # X_cumulative = np.vstack((constant, X_cumulative))
                    # # # trainX = X_cumulative[:, :int(0.5*len(cts))]
                    # # # testX = X_cumulative[:, int(0.5*len(cts)):]
                    # # # trainY = cts[:int(0.5*len(cts))]
                    # # # testY = cts[int(0.5*len(cts)):]
                    # # print(X_cumulative.shape, cts.shape)
                    # # # fit model
                    # # # model = sm.GLM(trainY, trainX.T, family=sm.families.Poisson(link=sm.families.links.log()))
                    # # model = sm.GLM(cts, X_cumulative.T, family=sm.families.Poisson(link=sm.families.links.log()))
                    # # model_res = model.fit()
                    # # # predict
                    # # train_pred = model_res.predict(X_cumulative.T)
                    # # # test_pred = model_res.predict(testX.T)

                    # # # r2 score
                    # # train_score = r2_score(trainY, train_pred)
                    # # # test_score = r2_score(testY, test_pred)
                    # # # beta coefficients
                    # # beta = model_res.params
                    # # # beta = beta[1:]
                    # # # rsquared
                    # # deviance = model_res.deviance
                    # # null_deviance = model_res.null_deviance
                    # # rsquared = (null_deviance - deviance)/null_deviance
                    # # # DEV explained
                    # # dev_explained = 1 - (deviance/null_deviance)
                    # # # dfbetas
                    # # # dfbetas = model_res.get_influence().dfbetas
                    # # # dfbetas = dfbetas[:,-1]
                    # # # absolute value of dfbetas
                    # # dfbetas = np.abs(dfbetas)
                    # # # residuals
                    # # residuals = model_res.resid_response
                    # # residuals = np.abs(residuals)


                    # # # plot
                    # # fig = plt.figure(figsize=(8,3))
                    # # ax = plt.subplot(1,1,1)
                    # # ax2 = ax.twinx()
                    # # ax2.plot(train_pred, color='red')
                    # # ax.plot(cts, color='blue')
                    # # ax.set_title('R2: ' + str(train_score) + ' RSquared: ' + str(rsquared) + ' r2Pred: ' + str(dev_explained))
                    # # plt.show()


                    # # # # plot residuals
                    # # # fig = plt.figure(figsize=(8,3))
                    # # # ax = plt.subplot(1,1,1)
                    # # # ax.plot(residuals, color='k')
                    # # # plt.show()

                    # # # # plot dfbetas
                    # # # fig = plt.figure(figsize=(8,3))
                    # # # ax = plt.subplot(1,1,1)
                    # # # ax.plot(dfbetas, color='k')
                    # # # plt.show()

                    # # pvalues = model_res.pvalues
                    # # pvalues = pvalues[:]

                    # # # plot betas
                    # # fig = plt.figure(figsize=(8,3))
                    # # ax = plt.subplot(1,1,1)
                    # # ax.plot(beta, color='k')
                    # # # highlights significant betas in red
                    # # idx = np.where(pvalues < 0.05)[0]
                    # # beta_copy = beta.copy()
                    # # beta_copy[idx] = np.nan
                    # # ax.plot(beta_copy, color='red')
                    # # plt.show()

            

                    # # # # plot imshow of X after regression fit
                    # # # fig = plt.figure(figsize=(8,3))
                    # # # # get 2d map of same shape as X using residual at each timepoint
                    # # # # residual_map = np.zeros(X.shape)
                    # # # # for i in range(X.shape[0]):
                    # # # #     residual_map[i,:] = X[i,:] - train_pred[i]
                    # # # # plt.imshow(residual_map, aspect='auto')
                    # # # plt.imshow(dfbetas_agg, aspect='auto', cmap='jet')
                    # # # plt.show()
                    # # # # gaussian smooth the 2d map of dfbetas
                    # # # fig = plt.figure(figsize=(8,3))
                    # # # from scipy.ndimage import gaussian_filter
                    # # # dfbetas_agg = gaussian_filter(dfbetas_agg, sigma=3)
                    # # # plt.imshow(dfbetas_agg, aspect='auto', cmap='jet')
                    # # # plt.show()

                    # # # fig = plt.figure(figsize=(8,3))
                    # # # plt.imshow(residuals_agg, aspect='auto', cmap='jet')
                    # # # plt.show()

    return ses_wins, ses_cts, ses_ids, ses_ctypes



def generate_time_ramps(num_bins, window):
    full_ramp = []
    scaled_full_ramp = []
    norm_full_ramp = []
    nmb_ramps = num_bins/window 
    # if nmb_ramps is not whole number, then round up to nearest whole number
    if nmb_ramps % 1 != 0:
        nmb_ramps = int(nmb_ramps) + 1
    for i in range(int(nmb_ramps)):
        part_ramp = np.arange(0,window+1,1)
        full_ramp.append(part_ramp)
        # scaled ramp should go from 0 to 1
        scaled_part_ramp = np.arange(0, 1+1/window, 1/window)
        scaled_full_ramp.append(scaled_part_ramp)
        # norm ramp should use linalg norm
        norm_part_ramp = part_ramp/np.linalg.norm(part_ramp)
        norm_full_ramp.append(norm_part_ramp)
    ramp_values = np.hstack(full_ramp)
    scaled_ramp_values = np.hstack(scaled_full_ramp)
    norm_ramp_values = np.hstack(norm_full_ramp)
    # cut to be same legnth as num bins
    ramp_values = ramp_values[:num_bins]
    norm_ramp_values = norm_ramp_values[:num_bins]
    scaled_ramp_values = scaled_ramp_values[:num_bins]
    return ramp_values, scaled_ramp_values, norm_ramp_values

def chebyshev_poly(degree, x):
    if degree == 0:
        return 1
    elif degree == 1:
        return x
    else:
        return 2 * x * chebyshev_poly(degree-1, x) - chebyshev_poly(degree-2, x)


def generate_cosine_basis(num_bins, T, win_ct):
    # nmb_ramps = num_bins / window

    # # If nmb_ramps is not a whole number, round up to the nearest whole number
    # if nmb_ramps % 1 != 0:
    #     nmb_ramps = int(nmb_ramps) + 1

    # freq = win_ct
    # t = np.linspace(0, freq, num_bins)
    # ramp_values = np.cos(t)

    cheb_poly = chebyshev_poly(win_ct, np.linspace(-1, 1, num_bins))
    ramp_values = cheb_poly



    # ramp_values = np.cos(2 * np.pi * t * freq)
    # Scaled ramp should go from 0 to 1
    scaled_ramp_values = (ramp_values - np.min(ramp_values)) / (np.max(ramp_values) - np.min(ramp_values))
    # Norm ramp should use linalg norm
    norm_ramp_values = ramp_values / np.linalg.norm(ramp_values)

    

    # frequency = 1/freq 
    # period = 1/frequency
    # phase = t[0] * (2 * np.pi / period)
    # print(frequency, period, phase)

    ramp_values = np.array(ramp_values)[:num_bins]
    scaled_ramp_values = np.array(scaled_ramp_values)[:num_bins]
    norm_ramp_values = np.array(norm_ramp_values)[:num_bins]

    return ramp_values, scaled_ramp_values, norm_ramp_values
   

def event_times_to_count(event_times, T, dt):
    # dt = settings_dict['bin_timestep']
    new_time_index = np.arange(0,T+dt,dt)
    ct, bins = np.histogram(event_times, bins=new_time_index)
    ct = np.asarray(ct)
    # norm_ct = (ct/dt - np.mean(ct/dt))/np.std(ct/dt)
    # norm_ct[norm_ct < 0] = 0 # cant have negative count

    return ct, bins[:-1]




if __name__ == '__main__':
    main()
