import os

def get_files_in_dir(dir_path):
    """
    Get all files in a directory.
    """
    return [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]