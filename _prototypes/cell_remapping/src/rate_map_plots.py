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
from _prototypes.cell_remapping.src.masks import flat_disk_mask

def plot_rate_remapping(prev, curr, plot_settings, data_dir):

    fig = TemplateFig()

    fig.density_plot(prev, fig.ax['1'])
    fig.density_plot(curr, fig.ax['2'])

    prev_key, curr_key = plot_settings['session_ids'][-1]
    wass = plot_settings['wass'][-1]
    unit_id = plot_settings['unit_id'][-1]
    name = plot_settings['name'][-1]

    title = prev_key + ' & ' + curr_key + ' : ' + str(wass)

    fig.f.suptitle(title, ha='center', fontweight='bold')

    """ save """
    # create a dsave and an fprefix
    # save_dir = PROJECT_PATH + '/_prototypes/cell_remapping/output/rate'
    save_dir = data_dir + '/output/regular'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fprefix = 'ratemap_cell_{}_{}_{}_unit_{}'.format(name, prev_key, curr_key, unit_id)

    ftemplate_short = "{}.{}"
    fshort = ftemplate_short.format(fprefix, 'pdf')
    fp = os.path.join(save_dir, fshort)
    fig.f.savefig(fp, dpi=360.)
    plt.close(fig.f)

def plot_obj_remapping(obj_rate_map, ses_rate_map, plot_settings, data_dir):

    fig = TemplateFig()

    fig.density_plot(obj_rate_map, fig.ax['1'])
    fig.density_plot(ses_rate_map, fig.ax['2'])

    ses_key = plot_settings['session_id'][-1]
    object_location = plot_settings['object_location'][-1]
    sliced_wass = plot_settings['obj_wass_'+str(object_location)][-1]
    unit_id = plot_settings['unit_id'][-1]
    name = plot_settings['name'][-1]

    if type(sliced_wass) == list:
        sliced_wass = sliced_wass[0]
    title = ses_key + ' & object ' + str(object_location) + ' : ' + str(round(sliced_wass, 2))
    # print(title)

    fig.f.suptitle(title, ha='center', fontweight='bold')

    """ save """
    # create a dsave and an fprefix
    # save_dir = PROJECT_PATH + '/_prototypes/cell_remapping/output/object'
    save_dir = data_dir + '/output/object'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fprefix = 'obj_ratemap_cell_{}_{}_{}_unit_{}'.format(name, ses_key, object_location, unit_id)

    ftemplate_short = "{}.{}"
    fshort = ftemplate_short.format(fprefix, 'pdf')
    fp = os.path.join(save_dir, fshort)
    fig.f.savefig(fp, dpi=360.)
    plt.close(fig.f)

def plot_fields_remapping(label_s, label_t, spatial_spike_train_s, spatial_spike_train_t, centroid_s, centroid_t, plot_settings, data_dir, settings, cylinder=False):

    target_rate_map_obj = spatial_spike_train_t.get_map('rate')
    target_map, _ = target_rate_map_obj.get_rate_map(new_size = settings['ratemap_dims'][0])

    y, x = target_map.shape

    source_rate_map_obj = spatial_spike_train_s.get_map('rate')
    source_map, _ = source_rate_map_obj.get_rate_map(new_size = settings['ratemap_dims'][0])
    

    if cylinder:
        source_map = flat_disk_mask(source_map)
        target_map = flat_disk_mask(target_map)
        label_s = flat_disk_mask(label_s)
        label_t = flat_disk_mask(label_t)


    fig = FieldsTemplateFig()

    fig.density_field_plot(source_map, centroid_s, fig.ax['1'])
    fig.density_field_plot(target_map, centroid_t, fig.ax['2'])
    fig.label_field_plot(label_s, centroid_s, fig.ax['3'])
    fig.label_field_plot(label_t, centroid_t, fig.ax['4'])
    fig.binary_field_plot(label_s, centroid_s, fig.ax['5'])
    fig.binary_field_plot(label_t, centroid_t, fig.ax['6'])

    prev_key, curr_key = plot_settings['session_ids'][-1]
    cumulative_wass = plot_settings['cumulative_wass'][-1]
    unit_id = plot_settings['unit_id'][-1]
    name = plot_settings['name'][-1]

    title = prev_key + ' & ' + curr_key + ' : ' + str(cumulative_wass)

    fig.f.suptitle(title, ha='center', fontweight='bold')
    # print(title)

    # fig.f.suptitle(title, ha='center', fontweight='bold', fontsize='large')

    """ save """
    # create a dsave and an fprefix
    # save_dir = PROJECT_PATH + '/_prototypes/cell_remapping/output/centroid'
    save_dir = data_dir + '/output/centroid'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fprefix = 'fields_ratemap_cell_{}_{}_{}_unit_{}'.format(name, prev_key, curr_key, unit_id)

    ftemplate_short = "{}.{}"
    fshort = ftemplate_short.format(fprefix, 'pdf')
    fp = os.path.join(save_dir, fshort)
    fig.f.savefig(fp, dpi=360.)
    plt.close(fig.f)

class FieldsTemplateFig():
    def __init__(self):
        self.f = plt.figure(figsize=(10, 18))
        # mpl.rc('font', **{'size': 20})


        self.gs = {
            'all': gridspec.GridSpec(3, 2, left=0.05, right=0.95, bottom=0.1, top=0.9, figure=self.f),
        }

        self.ax = {
            '1': self.f.add_subplot(self.gs['all'][:1, :1]),
            '2': self.f.add_subplot(self.gs['all'][:1, 1:2]),
            '3': self.f.add_subplot(self.gs['all'][1:2, :1]),
            '4': self.f.add_subplot(self.gs['all'][1:2, 1:2]),
            '5': self.f.add_subplot(self.gs['all'][2:3, :1]),
            '6': self.f.add_subplot(self.gs['all'][2:3, 1:2]),
        }

    def density_field_plot(self, rate_map, centroids, ax):

        # toplot = _interpolate_matrix(rate_map, new_size=(256,256), cv2_interpolation_method=cv2.INTER_NEAREST)
        toplot = rate_map

        # img = ax.imshow(np.uint8(cm.jet(toplot)*255))
        img = ax.imshow(toplot, cmap='jet')

        for c in centroids:
            ax.plot(c[1], c[0], 'r.', markersize=10)

        self.f.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    def label_field_plot(self, labels, centroids, ax):

        # toplot = _interpolate_matrix(labels, new_size=(256,256), cv2_interpolation_method=cv2.INTER_NEAREST)
        toplot = labels

        img = ax.imshow(toplot, cmap='Greys_r')

        for c in centroids:
            ax.plot(c[1], c[0], 'r.', markersize=10)

    def binary_field_plot(self, labels, centroids, ax):

        labels[labels != labels] = 0

        labels[labels != 0] = 1

        # toplot = _interpolate_matrix(labels, new_size=(256,256), cv2_interpolation_method=cv2.INTER_NEAREST)
        toplot = labels

        img = ax.imshow(toplot, cmap='Greys')

        for c in centroids:
            ax.plot(c[1], c[0], 'r.', markersize=10)



class TemplateFig():
    def __init__(self):
        self.f = plt.figure(figsize=(10, 6))
        # mpl.rc('font', **{'size': 20})


        self.gs = {
            'all': gridspec.GridSpec(1, 2, left=0.05, right=0.95, bottom=0.1, top=0.9, figure=self.f),
        }

        self.ax = {
            '1': self.f.add_subplot(self.gs['all'][:, :1]),
            '2': self.f.add_subplot(self.gs['all'][:, 1:2]),
        }

    def density_plot(self, rate_map, ax):

        # toplot = _interpolate_matrix(rate_map, new_size=(256,256), cv2_interpolation_method=cv2.INTER_NEAREST)
        toplot = rate_map

        # img = ax.imshow(np.uint8(cm.jet(toplot)*255))
        img = ax.imshow(toplot, cmap='jet')

        self.f.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

