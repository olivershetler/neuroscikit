# import os
# import sys
# import numpy as np

# PROJECT_PATH = os.getcwd()
# sys.path.append(PROJECT_PATH)

# from core.core_utils import make_seconds_index_from_rate
# from library.cluster import feature_wave_PCX, wave_PCA, create_features, feature_energy, isolation_distance, L_ratio, mahal
# from library.batch_space import SpikeClusterBatch
# from core.spikes import SpikeCluster

# def make_1D_timestamps(T=2, dt=0.02):
#     time = np.arange(0,T,dt)

#     spk_count = np.random.choice(len(time), size=1)
#     while spk_count <= 10:
#         spk_count = np.random.choice(len(time), size=1)
#     spk_time = np.random.choice(time, size=spk_count, replace=False).tolist()

#     return spk_time

# def make_2D_arena(count=100):
#     return np.random.sample(count), np.random.sample(count)

# def make_waveforms(channel_count, spike_count, samples_per_wave):
#     waveforms = np.zeros((channel_count, spike_count, samples_per_wave))

#     for i in range(channel_count):
#         for j in range(samples_per_wave):
#             waveforms[i,:,j] = np.random.randint(-20,20,size=spike_count).tolist()

#     return waveforms.tolist()

# def make_clusters(timestamps, cluster_count):
#     cluster_labels = []
#     for i in range(len(timestamps)):
#         idx = np.random.choice(cluster_count, size=1)[0]
#         cluster_labels.append(int(idx))
#     return cluster_labels

# def make_spike_cluster_batch():
#     event_times = make_1D_timestamps()
#     ch_count = 4
#     samples_per_wave = 50
#     waveforms = make_waveforms(ch_count, len(event_times), samples_per_wave)
#     cluster_count = 10

#     event_labels = make_clusters(event_times, cluster_count)

#     T = 2
#     dt = .02

#     input_dict1 = {}
#     input_dict1['duration'] = int(T)
#     input_dict1['sample_rate'] = float(1 / dt)
#     input_dict1['event_times'] = event_times
#     input_dict1['event_labels'] = event_labels


#     for i in range(ch_count):
#         key = 'channel_' + str(i+1)
#         input_dict1[key] = waveforms[i]

#     spike_cluster_batch = SpikeClusterBatch(input_dict1)

#     return spike_cluster_batch

# def make_spike_cluster():
#     event_times = make_1D_timestamps()
#     ch_count = 8
#     samples_per_wave = 50
#     waveforms = make_waveforms(ch_count, len(event_times), samples_per_wave)

#     T = 2
#     dt = .02
#     idx = np.random.choice(len(event_times), size=1)[0]

#     input_dict1 = {}
#     input_dict1['duration'] = int(T)
#     input_dict1['sample_rate'] = float(1 / dt)
#     input_dict1['event_times'] = event_times
#     input_dict1['cluster_label'] = int(idx + 1)


#     for i in range(ch_count):
#         key = 'channel_' + str(i+1)
#         input_dict1[key] = waveforms[i]

#     spike_cluster = SpikeCluster(input_dict1)

#     return spike_cluster

# def test_wave_PCA():
#     cv = np.eye(5)

#     pc, rpc, ev, rev = wave_PCA(cv)

#     assert type(pc) == np.ndarray
#     assert type(rpc) == np.ndarray
#     assert type(ev) == np.ndarray
#     assert type(rev) == np.ndarray

# def test_feature_wave_PCX():
#     spike_cluster = make_spike_cluster()
#     spike_cluster_batch = make_spike_cluster_batch()
    
#     wavePCData = feature_wave_PCX(spike_cluster)
#     wavePCData_batch = feature_wave_PCX(spike_cluster_batch)

#     assert type(wavePCData) == np.ndarray
#     assert type(wavePCData_batch) == np.ndarray

# def test_create_features():
#     spike_cluster = make_spike_cluster_batch()
#     FD = create_features(spike_cluster)
#     assert type(FD) == np.ndarray

#     spike_cluster_batch = make_spike_cluster_batch()
#     FD = create_features(spike_cluster_batch)
#     assert type(FD) == np.ndarray

# def test_feature_energy():
#     spike_cluster = make_spike_cluster()
#     spike_cluster_batch = make_spike_cluster_batch()

#     E = feature_energy(spike_cluster)
#     assert type(E) == np.ndarray

#     E = feature_energy(spike_cluster_batch)
#     assert type(E) == np.ndarray

# def test_isolation_distance():
#     spike_cluster = make_spike_cluster()
#     spike_cluster_batch = make_spike_cluster_batch()

#     iso_dist = isolation_distance(spike_cluster)
#     assert type(iso_dist) == float

#     iso_dist = isolation_distance(spike_cluster_batch)
#     assert type(iso_dist) == float

# def test_L_ratio():
#     spike_cluster = make_spike_cluster()
#     spike_cluster_batch = make_spike_cluster_batch()

#     L, ratio, df = L_ratio(spike_cluster)

#     assert type(L) == np.float64
#     assert type(ratio) == np.float64 
#     assert type(df) == int 

#     L, ratio, df = L_ratio(spike_cluster_batch)

#     assert type(L) == np.float64
#     assert type(ratio) == np.float64 
#     assert type(df) == int 

# def test_mahal():
#     spike_cluster = make_spike_cluster()
#     spike_cluster_batch = make_spike_cluster_batch()

#     FD = create_features(spike_cluster)
#     FD_batch = create_features(spike_cluster_batch)

#     d = mahal(FD, FD[spike_cluster.cluster_labels, :])
#     assert type(d) == np.ndarray

#     d = mahal(FD_batch, FD_batch[spike_cluster_batch.cluster_labels, :])
#     assert type(d) == np.ndarray

# if __name__ == '__main__':
#     # test_feature_wave_PCX()
#     # test_feature_energy()
#     # test_wave_PCA()
#     test_create_features()
#     # test_mahal()
#     # test_isolation_distance()
#     # test_L_ratio()




