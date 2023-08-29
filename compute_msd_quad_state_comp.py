import os

import numpy as np
import pandas as pd

from typer import run
from pkg_resources import resource_filename

from pymor.core.logger import getLogger

from quad_ph_mor.experiments.msd import MSDExperiment
from quad_ph_mor.experiments.methods import FrequencyIODMD, IODMD, OI, FrequencyOI, PHDMD, FrequencyPHDMD
from quad_ph_mor.tools.misc import product_dict


FILEPATH = resource_filename('quad_ph_mor', 'experiments/data/msd/quad_state_comp')


def main():
    logger = getLogger('pymor')
    logger.setLevel('ERROR')

    msd = MSDExperiment(log_to_file=True)
    msd.setup(order=100)
    red_orders = [ k for k in range(1, 4) ]
    for order in red_orders:
        msd.reduce(order)
    control = [ '[(t[0] < .5) * 1., (t[0] < .5) * -1.]' ]
    control_expr = [ lambda t: np.vstack([np.where(t < .5, 1., 0.), np.where(t < .5, -1., 0.)]) ]
    nt = [ 10000 ]
    initial_alpha = [ .1 ]

    phdmd_args = {
        'experiment': [ msd ],
        'order': red_orders,
        'control': control,
        'control_expr': control_expr,
        'nt': nt,
        'initial_alpha': initial_alpha,
    }

    meth_args = [
        (FrequencyPHDMD(), phdmd_args),
        (PHDMD(), phdmd_args),
    ]

    msd.logger.info('Running comp MSD...')

    columns = ['method', 'ord', 'l_time', 'q_time', 'control']
    df = pd.DataFrame(columns=columns)

    for n_meth, (method, kwargs) in enumerate(meth_args):
        num_args = len(list(product_dict(**kwargs)))

        for index, arg in enumerate(product_dict(**kwargs)):
            msd.logger.info(f'Running args {n_meth * num_args + index + 1} of {len(meth_args) * num_args}!')
            msd.logger.debug(f'ARGS: {arg}')

            result = method(arg)
            save_arg = arg
            save_arg.pop('control_expr', None)
            result['args'] = save_arg
            result['method'] = method.__NAME__
            results = [result]

            if result['success']:
                msd.logger.debug(f"Model finished {result['method']} in {result['lin_time']} + {result['quad_time']} seconds!")
            else:
                msd.logger.debug(f"Model failed {result['method']} in {result['lin_time']} + {result['quad_time']} seconds!")

            msd.logger.debug(f'Saving results...')
            ndf = pd.DataFrame([
                [result['method'], result['order'], result['lin_time'], result['quad_time'], 'jump']
            ], columns=columns)
            df = pd.concat([df, ndf], ignore_index=True)

    control = [ '[exp(-t[0] / 2) * sin(t[0]**2), exp(-t[0] / 2) * cos(t[0]**2)]' ]
    control_expr = [ lambda t: np.array([np.exp(-t**2 / 2) * np.sin(t**2), np.exp(-t / 2) * np.cos(t**2)]) ]

    phdmd_args = {
        'experiment': [ msd ],
        'order': red_orders,
        'control': control,
        'control_expr': control_expr,
        'nt': nt,
        'initial_alpha': initial_alpha,
    }

    meth_args = [
        (FrequencyPHDMD(), phdmd_args),
        (PHDMD(), phdmd_args),
    ]

    for n_meth, (method, kwargs) in enumerate(meth_args):
        num_args = len(list(product_dict(**kwargs)))

        results = []
        for index, arg in enumerate(product_dict(**kwargs)):
            msd.logger.info(f'Running args {n_meth * num_args + index + 1} of {len(meth_args) * num_args}!')
            msd.logger.debug(f'ARGS: {arg}')

            result = method(arg)
            save_arg = arg
            save_arg.pop('control_expr', None)
            result['args'] = save_arg
            result['method'] = method.__NAME__
            results.append(result)

            if result['success']:
                msd.logger.debug(f"Model finished {result['method']} in {result['lin_time']} + {result['quad_time']} seconds!")
            else:
                msd.logger.debug(f"Model failed {result['method']} in {result['lin_time']} + {result['quad_time']} seconds!")

            msd.logger.debug(f'Saving results...')
            ndf = pd.DataFrame([
                [result['method'], result['order'], result['lin_time'], result['quad_time'], 'sine']
            ], columns=columns)
            df = pd.concat([df, ndf], ignore_index=True)

    df = df.replace([np.inf, -np.inf], np.nan)
    quad_file = os.path.join(FILEPATH, 'quad_state_comp.csv')
    if not os.path.exists(os.path.dirname(quad_file)):
        os.makedirs(os.path.dirname(quad_file), exist_ok=True)
    df.to_csv(quad_file, index=False)


if __name__ == '__main__':
    run(main)
