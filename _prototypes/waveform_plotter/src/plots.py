import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)


def plot_cell_waveform(cell, data_dir):

    fig = TemplateFig()

    for i in range(4):
        ch = cell.signal[:,i,:]
        idx = np.random.choice(len(ch), size=200)
        waves = ch[idx, :]
        avg_wave = np.mean(ch, axis=0)

        fig.waveform_channel_plot(waves, avg_wave, str(i+1), fig.ax[str(i+1)])

    animal = cell.animal_id
    session = cell.ses_key
    unit = cell.cluster.cluster_label

    title = str(animal) + '_' + str(session) + '_unit_' + str(unit)

    fig.f.suptitle(title, ha='center', fontweight='bold', fontsize='large')

    """ save """
    # create a dsave and an fprefix
    save_dir = data_dir + r'/output/' + str(animal) + r'/' + str(session) + r'/'
    fprefix = r'waveform_{}_{}_unit_{}'.format(animal, session, unit)

    ftemplate_short = "{}.{}"
    fshort = ftemplate_short.format(fprefix, r'pdf')
    fp = os.path.join(save_dir, fshort)
    fig.f.savefig(fp, dpi=360.)
    plt.close(fig.f)


class TemplateFig():
    def __init__(self):
        self.f = plt.figure(figsize=(12, 6))
        # mpl.rc('font', **{'size': 20})


        self.gs = {
            'all': gridspec.GridSpec(1, 4, left=0.05, right=0.95, bottom=0.1, top=0.85, figure=self.f),
        }

        self.ax = {
            '1': self.f.add_subplot(self.gs['all'][:, :1]),
            '2': self.f.add_subplot(self.gs['all'][:, 1:2]),
            '3': self.f.add_subplot(self.gs['all'][:, 2:3]),
            '4': self.f.add_subplot(self.gs['all'][:, 3:]),
        }

    def waveform_channel_plot(self, waveforms, avg_waveform, channel, ax):

        ax.plot(waveforms.T, color='grey')

        ax.plot(avg_waveform, c='k', lw=2)

        ax.set_title('Channel ' + str(int(channel)))





