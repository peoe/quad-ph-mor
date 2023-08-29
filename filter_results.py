import os

import numpy as np
import pandas as pd

from itertools import product
from pkg_resources import resource_filename
from typer import run


FILEPATH = resource_filename('quad_ph_mor', 'experiments/data')
NAMES = [
    'msd',
    'wave1d',
]
INITIALS = [
    'jump',
    'sine',
]
ERROR = '{name}_{initial}_errors.csv'
ENERGY = '{name}_{initial}_energies.csv'


def main():
    # filter error and energy files
    num = len(NAMES) * len(INITIALS)
    for i, (name, initial) in enumerate(product(NAMES, INITIALS)):
        print(f'Running {i + 1} of {num} with name {name} and initial {initial}')

        errors = ERROR.format(name=name, initial=initial)

        err_df = pd.read_csv(os.path.join(FILEPATH, f'{name}/{initial}/csv/' + errors))
        err_df = err_df.replace([np.inf, -np.inf], np.nan)

        phdmd_file = os.path.join(FILEPATH, f'{name}/{initial}/csv/{name}_phdmd_err.csv')
        phdmd_err = err_df[err_df['method'] == 'FrequencyPHDMD'].filter(items=['ord', 'l_time', 'q_time', 'FOMh2', 'PHIRKAh2', 'rPHIRKAh2', 'lh2', 'rlh2', 'qh2', 'rqh2', 'FOMhinf', 'PHIRKAhinf', 'rPHIRKAhinf', 'lhinf', 'rlhinf', 'qhinf', 'rqhinf']).loc[err_df['ord'].isin(list(range(2, 21)))]
        phdmd_err.to_csv(phdmd_file, index=False)

        iodmd_file = os.path.join(FILEPATH, f'{name}/{initial}/csv/{name}_iodmd_err.csv')
        iodmd_err = err_df[err_df['method'] == 'FrequencyIODMD'].filter(items=['ord', 'l_time', 'q_time', 'FOMh2', 'PHIRKAh2', 'rPHIRKAh2', 'lh2', 'rlh2', 'qh2', 'rqh2', 'FOMhinf', 'PHIRKAhinf', 'rPHIRKAhinf', 'lhinf', 'rlhinf', 'qhinf', 'rqhinf']).loc[err_df['ord'].isin(list(range(2, 21)))]
        iodmd_err.to_csv(iodmd_file, index=False)

        oi_file = os.path.join(FILEPATH, f'{name}/{initial}/csv/{name}_oi_err.csv')
        oi_err = err_df[err_df['method'] == 'FrequencyOI'].filter(items=['ord', 'l_time', 'q_time', 'FOMh2', 'PHIRKAh2', 'rPHIRKAh2', 'lh2', 'rlh2', 'qh2', 'rqh2', 'FOMhinf', 'PHIRKAhinf', 'rPHIRKAhinf', 'lhinf', 'rlhinf', 'qhinf', 'rqhinf']).loc[err_df['ord'].isin(list(range(2, 21)))]
        oi_err.to_csv(oi_file, index=False)

    # filter comparison of quad i/o and quad state phdmd runs
    truncate = lambda flt: float(f'{flt:1.5f}')
    state_file = os.path.join(FILEPATH, 'msd/quad_state_comp/quad_state_comp.csv')
    state_df = pd.read_csv(state_file)
    state_df = state_df.replace([np.inf, -np.inf], np.nan)
    state_df['ltime'] = state_df['l_time'].map(truncate)
    state_df = state_df.replace([np.inf, -np.inf], np.nan)
    state_df['qtime'] = state_df['q_time'].map(truncate)
    state_df.sort_values(['ord', 'method']).to_csv(state_file, index=False)


if __name__ == '__main__':
    run(main)
