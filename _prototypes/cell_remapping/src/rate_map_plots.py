import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import matplotlib as mpl
import cv2

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.map_utils import _interpolate_matrix

def plot_rate_remapping(prev, curr, plot_settings):

    fig = TemplateFig()

    fig.density_plot(prev, fig.ax['1'])
    fig.density_plot(curr, fig.ax['2'])

    prev_key, curr_key = plot_settings['session_ids'][-1]
    sliced_wass = plot_settings['sliced_wass'][-1]
    unit_id = plot_settings['unit_id'][-1]
    animal_id = plot_settings['animal_id'][-1]

    title = prev_key + ' & ' + curr_key + ' : ' + str(sliced_wass)

    fig.f.suptitle(title, ha='center', fontweight='bold')

    """ save """
    # create a dsave and an fprefix
    save_dir = PROJECT_PATH + '/_prototypes/cell_remapping/output/rate'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fprefix = 'ratemap_cell_{}_{}_{}_unit_{}'.format(animal_id, prev_key, curr_key, unit_id)

    ftemplate_short = "{}.{}"
    fshort = ftemplate_short.format(fprefix, 'pdf')
    fp = os.path.join(save_dir, fshort)
    fig.f.savefig(fp, dpi=360.)
    plt.close(fig.f)

def plot_obj_remapping(obj_rate_map, ses_rate_map, plot_settings):

    fig = TemplateFig()

    fig.density_plot(obj_rate_map, fig.ax['1'])
    fig.density_plot(ses_rate_map, fig.ax['2'])

    ses_key = plot_settings['session_id'][-1]
    object_location = plot_settings['object_location'][-1]
    sliced_wass = plot_settings['obj_wass_'+str(object_location)][-1]
    unit_id = plot_settings['unit_id'][-1]
    animal_id = plot_settings['animal_id'][-1]

    title = ses_key + ' & object ' + str(object_location) + ' : ' + str(round(sliced_wass, 2))
    # print(title)

    fig.f.suptitle(title, ha='center', fontweight='bold', fontsize='large')

    """ save """
    # create a dsave and an fprefix
    save_dir = PROJECT_PATH + '/_prototypes/cell_remapping/output/object'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fprefix = 'ratemap_cell_{}_{}_{}_unit_{}'.format(animal_id, ses_key, object_location, unit_id)

    ftemplate_short = "{}.{}"
    fshort = ftemplate_short.format(fprefix, 'pdf')
    fp = os.path.join(save_dir, fshort)
    fig.f.savefig(fp, dpi=360.)
    plt.close(fig.f)


class TemplateFig():
    def __init__(self):
        self.f = plt.figure(figsize=(10, 4))
        # mpl.rc('font', **{'size': 20})


        self.gs = {
            'all': gridspec.GridSpec(1, 2, left=0.05, right=0.95, bottom=0.05, top=0.95, figure=self.f),
        }

        self.ax = {
            '1': self.f.add_subplot(self.gs['all'][:, :1]),
            '2': self.f.add_subplot(self.gs['all'][:, 1:2]),
        }

    def density_plot(self, rate_map, ax):

        # toplot = _interpolate_matrix(rate_map, new_size=(256,256), cv2_interpolation_method=cv2.INTER_NEAREST)
        toplot = rate_map

        img = ax.imshow(np.uint8(cm.jet(toplot)*255))

        self.f.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

