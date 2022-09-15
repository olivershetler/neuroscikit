import os
import sys
import numpy as np

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from core.core_utils import make_seconds_index_from_rate
from library.spatial import speed2D

def make_1D_timestamps(T=2, dt=0.02):
    time = np.arange(0,T,dt)

    spk_count = np.random.choice(len(time), size=1)
    while spk_count <= 10:
        spk_count = np.random.choice(len(time), size=1)
    spk_time = np.random.choice(time, size=spk_count, replace=False).tolist()

    return spk_time

def make_2D_arena(count=100):
    return np.random.sample(count), np.random.sample(count)

def test_speed2D():
    T = 2
    dt = .02
    pos_t = make_seconds_index_from_rate(T, 1/dt)

    pos_x, pos_y = make_2D_arena(len(pos_t))
    v_convolved = speed2D(pos_x, pos_y, pos_t)

    assert type(v_convolved) == np.ndarray

if __name__ == '__main__':
    test_speed2D()


