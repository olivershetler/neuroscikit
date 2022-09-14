import os
import sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
 

from library.spatial.speed2D import speed2D


__all__ = ['speed2D']

if __name__ == '__main__':
    pass
