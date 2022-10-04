import os, sys

PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)

from library.study_space import Session
from x_io.rw.axona.batch_read import make_study

def read_sequential_sessions(dir1, settings_dict: dict, dir2=None):
    """
    Can input one folder with both sessions inside or two folders (one for each session)
    """
    if dir2 == None:
        assert os.path.isdir(dir1), 'input path is not a folder'
        study = make_study(dir1, settings_dict)
    else:
        assert os.path.isdir(dir1) and os.path.isdir(dir2), 'input paths are not folders'
        study = make_study([dir1, dir2], settings_dict)

    study.make_animals()
    animals = study.animals
    assert len(animals) == 1
    assert len(animals[0].sessions) == 2
    session_1  = animals[0].sessions['session_1']
    session_2  = animals[0].sessions['session_2']

    return session_1, session_2 
