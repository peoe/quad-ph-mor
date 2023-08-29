import numpy as np
from pkg_resources import resource_filename
import os
import pickle


FILEPATH = resource_filename('quad_ph_mor', 'experiments/data')


def load(filename, format='.dat'):
    file_path = os.path.join(FILEPATH, filename + format)
    try:
        with open(file_path, 'rb') as file:
            objs = []
            while True:
                try:
                    objs.append(pickle.load(file))
                except EOFError:
                    break

        assert len(objs) == 1
        return objs[0]
    except FileNotFoundError:
        raise OSError(f'No data found under the file path: {file_path}!')


def save(filename, data, format='.dat'):
    file_path = os.path.join(FILEPATH, filename + format)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(os.path.join(FILEPATH, filename + format), 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def savez(filename, **data):
    file_path = os.path.join(FILEPATH, filename)
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(os.path.join(FILEPATH, filename), 'wb') as file:
        np.savez(file, **data)
