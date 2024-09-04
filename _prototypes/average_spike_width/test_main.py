import os, sys
PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
# Set necessary paths / make project path = ...../neuroscikit/
parent_dir = os.path.dirname(PROJECT_PATH)
sys.path.append(parent_dir)
data_dir = parent_dir + r'\neuroscikit_test_data\sequential_axona_sessions'


from main import main
from src.settings import settings_dict

def test_main():
    main()