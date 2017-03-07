import os


def procses_data(data_path):
    pass


def split_data(data_path):
    pass


def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass


def get_batch():
    pass
