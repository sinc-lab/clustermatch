import os


def get_data_file(data_filename):
    directorio = os.path.dirname(os.path.abspath(__file__))
    directorio = os.path.join(directorio, 'data/')
    return os.path.join(directorio, data_filename)
