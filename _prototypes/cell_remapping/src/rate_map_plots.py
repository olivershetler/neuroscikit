import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import matplotlib as mpl
import statsmodels.api as sm
import cv2

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.map_utils import _interpolate_matrix
from _prototypes.cell_remapping.src.masks import flat_disk_mask

def plot_regular_remapping(prev, curr, plot_settings, data_dir):

    fig = TemplateFig()

    fig.density_plot(prev, fig.ax['1'])
    fig.density_plot(curr, fig.ax['2'])

    prev_key, curr_key = plot_settings['session_ids'][-1]
    wass = plot_settings['whole_wass'][-1]
    unit_id = plot_settings['unit_id'][-1]
    name = plot_settings['name'][-1]
    tetrode = plot_settings['tetrode'][-1]

    title = prev_key + ' & ' + curr_key + ' : ' + str(wass)

    fig.f.suptitle(title, ha='center', fontweight='bold')

    """ save """
    # create a dsave and an fprefix
    # save_dir = PROJECT_PATH + '/_prototypes/cell_remapping/remapping_output/rate'
    save_dir = data_dir + '/remapping_output/regular'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fprefix = 'ratemap_cell_{}_tet_{}_ses_{}_{}_unit_{}'.format(name, tetrode, prev_key, curr_key, unit_id)

    ftemplate_short = "{}.{}"
    fshort = ftemplate_short.format(fprefix, 'pdf')
    fp = os.path.join(save_dir, fshort)
    fig.f.savefig(fp, dpi=360.)
    plt.close(fig.f)

def plot_shuffled_regular_remapping(prev, curr, ref_wass_dist, prev_sample, curr_sample, plot_settings, data_dir):

    print(np.array(prev).shape, np.array(curr).shape)

    fig = ShuffledTemplateFig()

    fig.histogram_plot(prev, fig.ax['1'])
    fig.histogram_plot(curr, fig.ax['2'])
    # fig.scatter_plot(prev, fig.ax['1'])
    # fig.scatter_plot(curr, fig.ax['2'])
    fig.histogram_plot(ref_wass_dist, fig.ax['3'])
    fig.qq_plot(ref_wass_dist, fig.ax['4'])

    # random map from shuffles
    chosen_prev = prev_sample[np.random.choice(np.arange(len(prev_sample)), 1)[0]]
    chosen_curr = curr_sample[np.random.choice(np.arange(len(curr_sample)), 1)[0]]

    fig.density_plot(chosen_prev, fig.ax['5'])
    fig.density_plot(chosen_curr, fig.ax['6'])

    prev_key, curr_key = plot_settings['session_ids'][-1]
    wass = plot_settings['whole_wass'][-1]
    unit_id = plot_settings['unit_id'][-1]
    name = plot_settings['name'][-1]
    tetrode = plot_settings['tetrode'][-1]

    title = prev_key + ' & ' + curr_key + ' : ' + str(wass)

    fig.f.suptitle(title, ha='center', fontweight='bold')

    """ save """
    # create a dsave and an fprefix
    # save_dir = PROJECT_PATH + '/_prototypes/cell_remapping/remapping_output/rate'
    save_dir = data_dir + '/remapping_output/regular_baseline'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fprefix = 'baseeline_cell_{}_tet_{}_ses_{}_{}_unit_{}'.format(name, tetrode, prev_key, curr_key, unit_id)

    ftemplate_short = "{}.{}"
    fshort = ftemplate_short.format(fprefix, 'pdf')
    fp = os.path.join(save_dir, fshort)
    fig.f.savefig(fp)
    plt.close(fig.f)

def plot_obj_remapping(obj_rate_map, ses_rate_map, labels, centroids, plot_settings, data_dir, cylinder=False):

    if cylinder:
        labels = flat_disk_mask(labels)

    fig = ObjectTemplateFig()

    fig.density_plot(obj_rate_map, fig.ax['1'])
    fig.density_plot(ses_rate_map, fig.ax['2'])
    fig.label_field_plot(labels, centroids, fig.ax['4'])
    fig.binary_field_plot(labels, centroids, fig.ax['3'])


    ses_key = plot_settings['session_id'][-1]
    object_location = plot_settings['object_location'][-1]
    sliced_wass = plot_settings['obj_wass_'+str(object_location)][-1]
    unit_id = plot_settings['unit_id'][-1]
    name = plot_settings['name'][-1]
    tetrode = plot_settings['tetrode'][-1]

    if type(sliced_wass) == list:
        sliced_wass = sliced_wass[0]
    title = ses_key + ' & object ' + str(object_location) + ' : ' + str(round(sliced_wass, 2))
    # print(title)

    fig.f.suptitle(title, ha='center', fontweight='bold')

    """ save """
    # create a dsave and an fprefix
    # save_dir = PROJECT_PATH + '/_prototypes/cell_remapping/remapping_output/object'
    save_dir = data_dir + '/remapping_output/object'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fprefix = 'obj_ratemap_cell_{}_tet_{}_ses_{}_{}_unit_{}'.format(name, tetrode, ses_key, object_location, unit_id)

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
    tetrode = plot_settings['tetrode'][-1]

    title = prev_key + ' & ' + curr_key + ' : ' + str(cumulative_wass)

    fig.f.suptitle(title, ha='center', fontweight='bold')
    # print(title)

    # fig.f.suptitle(title, ha='center', fontweight='bold', fontsize='large')

    """ save """
    # create a dsave and an fprefix
    # save_dir = PROJECT_PATH + '/_prototypes/cell_remapping/remapping_output/centroid'
    save_dir = data_dir + '/remapping_output/centroid'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    fprefix = 'fields_ratemap_cell_{}_tet_{}_ses_{}_{}_unit_{}'.format(name, tetrode, prev_key, curr_key, unit_id)

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

class ShuffledTemplateFig():
    def __init__(self):
        self.f = plt.figure(figsize=(10, 24))
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

    def histogram_plot(self, to_plot, ax):

        ct, bins = np.histogram(to_plot, bins=100)
        l1 = ax.bar(bins[:-1], ct, color='k', width=0.1, label='unnorm')

        ct, bins = np.histogram(to_plot/np.sum(to_plot), bins=100, density=True)
        ax2 = ax.twinx().twiny()
        l2 = ax2.bar(bins[:-1], ct, color='r', width=0.1, label='norm')

        lns = [l1, l2]
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)

    def qq_plot(self, to_plot, ax):

        sm.qqplot(np.asarray(to_plot), line='q', c='k', markersize=2, ax=ax)

    def density_plot(self, rate_map, ax):

        # toplot = _interpolate_matrix(rate_map, new_size=(256,256), cv2_interpolation_method=cv2.INTER_NEAREST)
        toplot = np.asarray(rate_map)

        # img = ax.imshow(np.uint8(cm.jet(toplot)*255))
        img = ax.imshow(toplot, cmap='jet')

        self.f.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    def scatter_plot(self, weights, ax):
        weights = np.asarray(weights)
        norm_weights = weights / np.max(weights)

        for w in np.arange(len(weights)):
            l1 = ax.scatter(np.arange(len(weights[w])), weights[w], c='k', label='unnorm')
        ax2 = ax.twinx()
        for w in np.arange(len(weights)):
            l2 = ax2.scatter(np.arange(len(norm_weights[w])), norm_weights[w], c='r', label='norm')

        lns = [l1, l2]
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs)

class ObjectTemplateFig():
    def __init__(self):
        self.f = plt.figure(figsize=(10, 6))
        # mpl.rc('font', **{'size': 20})


        self.gs = {
            'all': gridspec.GridSpec(2, 2, left=0.05, right=0.95, bottom=0.1, top=0.9, figure=self.f),
        }

        self.ax = {
            '1': self.f.add_subplot(self.gs['all'][:1, :1]),
            '2': self.f.add_subplot(self.gs['all'][:1, 1:2]),
            '3': self.f.add_subplot(self.gs['all'][1:2, :1]),
            '4': self.f.add_subplot(self.gs['all'][1:2, 1:2]),
        }

    def density_plot(self, rate_map, ax):

        # toplot = _interpolate_matrix(rate_map, new_size=(256,256), cv2_interpolation_method=cv2.INTER_NEAREST)
        toplot = rate_map

        # img = ax.imshow(np.uint8(cm.jet(toplot)*255))
        img = ax.imshow(toplot, cmap='jet')

        self.f.colorbar(img, ax=ax, fraction=0.046, pad=0.04)

    def label_field_plot(self, labels, centroids, ax):

        # toplot = _interpolate_matrix(labels, new_size=(256,256), cv2_interpolation_method=cv2.INTER_NEAREST)
        toplot = labels

        img = ax.imshow(toplot, cmap='Greys_r')

        for c in centroids:
            ax.plot(c[1], c[0], 'r.', markersize=10) 

        self.f.colorbar(img, ax=ax, fraction=0.046, pad=0.04)


    def binary_field_plot(self, labels, centroids, ax):

        labels[labels != labels] = 0

        labels[labels != 0] = 1

        # toplot = _interpolate_matrix(labels, new_size=(256,256), cv2_interpolation_method=cv2.INTER_NEAREST)
        toplot = labels

        img = ax.imshow(toplot, cmap='Greys')

        for c in centroids:
            ax.plot(c[1], c[0], 'r.', markersize=10)

        self.f.colorbar(img, ax=ax, fraction=0.046, pad=0.04)



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

