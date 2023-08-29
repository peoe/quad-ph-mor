import numpy as np

from typer import run

from pymor.core.logger import getLogger

from quad_ph_mor.experiments.wave import Wave1DExperiment
from quad_ph_mor.experiments.methods import FrequencyIODMD, IODMD, OI, FrequencyOI, PHDMD, FrequencyPHDMD
from quad_ph_mor.tools.misc import product_dict


def main():
    logger = getLogger('pymor')
    logger.setLevel('ERROR')

    wave = Wave1DExperiment(log_to_file=True)
    wave.setup(order=100, regularization=2e-3)
    red_orders = [ k for k in range(2, 10) ]
    for order in red_orders:
        wave.reduce(order)
    control = [ '[exp(-t[0] / 2) * sin(t[0]**2), exp(-t[0] / 2) * cos(t[0]**2)]' ]
    control_expr = [ lambda t: np.array([np.exp(-t**2 / 2) * np.sin(t**2), np.exp(-t / 2) * np.cos(t**2)]) ]
    disc_control = [ '[exp(-(t[0] / 100.) / 2) * sin((t[0] / 100.)**2), exp(-(t[0] / 100.) / 2) * cos((t[0] / 100.)**2)]' ]
    nt = [ 10000 ]
    initial_alpha = [ .1 ]

    phdmd_args = {
        'experiment': [ wave ],
        'order': red_orders,
        'control': control,
        'control_expr': control_expr,
        'nt': nt,
        'initial_alpha': initial_alpha,
    }

    iodmd_args = {
        'experiment': [ wave ],
        'order': red_orders,
        'control': control,
        'control_expr': control_expr,
        'disc_control': disc_control,
        'nt': nt,
    }

    oi_args = {
        'experiment': [ wave ],
        'order': red_orders,
        'control': control,
        'control_expr': control_expr,
        'nt': nt,
    }

    meth_args = [
        (FrequencyIODMD(), iodmd_args),
        (FrequencyOI(), oi_args),
        (FrequencyPHDMD(), phdmd_args),
    ]

    wave.logger.info('Running sine Wave...')

    for n_meth, (method, kwargs) in enumerate(meth_args):
        num_args = len(list(product_dict(**kwargs)))

        for index, arg in enumerate(product_dict(**kwargs)):
            wave.logger.info(f'Running args {n_meth * num_args + index + 1} of {len(meth_args) * num_args}!')
            wave.logger.debug(f'ARGS: {arg}')

            result = method(arg)
            save_arg = arg
            save_arg.pop('control_expr', None)
            result['args'] = save_arg
            result['method'] = method.__NAME__
            results = [result]

            if result['success']:
                wave.logger.debug(f"Model finished {result['method']} in {result['lin_time']} + {result['quad_time']} seconds!")
            else:
                wave.logger.debug(f"Model failed {result['method']} in {result['lin_time']} + {result['quad_time']} seconds!")

            wave.logger.debug('Saving results...')
            wave.save_data('sine', results=results)


if __name__ == '__main__':
    run(main)
