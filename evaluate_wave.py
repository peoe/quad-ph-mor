import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pkg_resources import resource_filename
from typer import run

from pymor.algorithms.timestepping import DiscreteTimeStepper
from pymor.core.logger import getLogger
from pymor.models.iosys import LTIModel

from quad_ph_mor.experiments.wave import Wave1DExperiment
from quad_ph_mor.tools.misc import to_lti
from quad_ph_mor.tools.timestepping import PHDMDImplicitMidpointTimeStepper


FILEPATH = resource_filename('quad_ph_mor', 'experiments/data/wave1d')
INITIALS = [
    'jump',
    'sine',
]
ERROR = 'wave1d_{initial}_errors.csv'
ENERGY = 'wave1d_{initial}_energies.csv'


def main():
    logger = getLogger('pymor')
    logger.setLevel('ERROR')

    wave = Wave1DExperiment(log_to_file=False)
    wave.setup(order=100)
    fom = wave.fom
    fom_h2 = fom.h2_norm()
    fom_hinf = fom.hinf_norm()

    for initial in INITIALS:
        columns = ['method', 'ord', 'l_time', 'q_time', 'FOMh2', 'PHIRKAh2', 'rPHIRKAh2', 'lh2', 'rlh2', 'qh2', 'rqh2', 'FOMhinf', 'PHIRKAhinf', 'rPHIRKAhinf', 'lhinf', 'rlhinf', 'qhinf', 'rqhinf']
        error_df = pd.DataFrame(columns=columns)

        T = 10.
        nt = 10000
        ts = np.linspace(0., T, nt + 1)

        if initial == 'jump':
            control = '[exp(-t[0] / 2) * sin(t[0]**2), exp(-t[0] / 2) * cos(t[0]**2)]'
            quad_control = '[exp(-t[0] / 2)**2 * sin(t[0]**2)**2, exp(-t[0] / 2)**2 * sin(t[0]**2) * cos(t[0]**2), exp(-t[0] / 2)**2 * cos(t[0]**2)**2]'
            disc_control = f'[exp(-(t[0] / {nt}) / 2) * sin((t[0] / {nt})**2), exp(-(t[0] / {nt}) / 2) * cos((t[0] / {nt})**2)]'
            quad_disc_control = f'[exp(-(t[0] / {nt}) / 2)**2 * sin((t[0] / {nt})**2)**2, exp(-(t[0] / {nt}) / 2)**2 * cos((t[0] / {nt})**2)**2, exp(-(t[0] / {nt}) / 2)**2 * sin((t[0] / {nt})**2) * cos((t[0] / {nt})**2)]'
        elif initial == 'sine':
            control = '[(t[0] < .5) * 1., (t[0] < .5) * -1.]'
            quad_control = '[(t[0] < .5) * 1., (t[0] < .5) * -1., (t[0] < .5) * 1.]'
            disc_control = f'[((t[0] / {nt}) < .5) * 1., ((t[0] / {nt}) < .5) * -1.]'
            quad_disc_control = f'[((t[0] / {nt}) < .5) * 1., ((t[0] / {nt}) < .5) * -1., ((t[0] / {nt}) < .5) * 1.]'
        else:
            raise NotImplementedError

        fenergy = None
        lenergies = {}
        qenergies = {}
        menergies = {}
        energy_columns = ['name', 'method', 'ord'] + [ f'{k}' for k in range(nt + 1) ]
        energy_df = pd.DataFrame(columns=energy_columns)

        target_dir = os.path.join(FILEPATH, initial)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir, exist_ok=True)
        for index, name in enumerate([ name for name in os.listdir(target_dir) if os.path.isfile(os.path.join(FILEPATH, f'{initial}/' + name)) ]):
            with open(os.path.join(FILEPATH, f'{initial}/' + name), 'rb') as file:
                try:
                    obj = pickle.load(file)
                except EOFError:
                    break

            wave.logger.info(f'Evaluating file {index}...')

            results = obj['results']
            for res in results:
                lin_success = res['lin_success'] if 'lin_success' in res else True
                quad_success = res['quad_success'] if 'quad_success' in res else True

                order = res['order']
                if order not in wave.rom_matrices:
                    wave.reduce(order)
                rom = wave.roms(order)

                method = res['method']

                if lin_success:
                    lrom = LTIModel.from_matrices(*res['lin_mats'])

                if quad_success:
                    if method in ('FrequencyPHDMD', 'FrequencySOBMOR', 'FrequencyOI', 'FrequencyIODMD'):
                        from pymor.operators.block import BlockColumnOperator, BlockDiagonalOperator, BlockRowOperator
                        from pymor.operators.constructions import ComponentProjectionOperator
                        from quad_ph_mor.tools.operator import QuadraticInputOperator

                        qrom = LTIModel.from_matrices(*res['quad_mats'])

                        B = BlockRowOperator([ lrom.B, qrom.B ])
                        C = BlockColumnOperator([ lrom.C, qrom.C ])
                        D = BlockDiagonalOperator([ lrom.D, qrom.D ])
                        q_op = QuadraticInputOperator(lrom.B.source)
                        c_op = ComponentProjectionOperator(range(lrom.B.source.dim), C.range)
                        B = B @ q_op
                        C = c_op @ C
                        D = c_op @ D @ q_op

                        A = lrom.A + qrom.A
                        E = lrom.E + qrom.E

                        model = LTIModel(A=A, B=B, C=C, E=E, D=D)
                    else:
                        qrom = LTIModel.from_matrices(*res['quad_mats'])
                        model = (lrom + qrom)

                nerror_df = pd.DataFrame([[
                        method,
                        res['order'],
                        res['lin_time'],
                        res['quad_time'],
                        fom_h2,
                        (fom - rom).h2_norm(),
                        (fom - rom).h2_norm() / fom_h2,
                        (fom - lrom).h2_norm() if lin_success else None,
                        (fom - lrom).h2_norm() / fom_h2 if lin_success else None,
                        (fom - model).h2_norm() if quad_success else None,
                        (fom - model).h2_norm() / fom_h2 if quad_success else None,
                        fom_hinf,
                        (fom - rom).hinf_norm(),
                        (fom - rom).hinf_norm() / fom_hinf,
                        (fom - lrom).hinf_norm() if lin_success else None,
                        (fom - lrom).hinf_norm() / fom_hinf if lin_success else None,
                        (fom - model).hinf_norm() if quad_success else None,
                        (fom - model).hinf_norm() / fom_hinf if quad_success else None,
                    ]],
                    columns=columns
                )
                error_df = pd.concat([error_df, nerror_df], ignore_index=True)

                if method == 'FrequencyIODMD':
                    plt.figure()
                    plt.title(f'Energy of {method} {order}')

                if fenergy is None:
                    flti = to_lti(fom).with_(D=None, T=T, time_stepper=PHDMDImplicitMidpointTimeStepper(nt=nt))
                    fsols = flti.solve(input=control)
                    fenergy = flti.E.pairwise_apply2(fsols, fsols)
                    fenergy_max = np.max(fenergy) * 1.1 # arbitrary cutoff
                    fenergy_min = np.min(fenergy) * 1.1 # arbitrary cutoff

                try:
                    if method not in lenergies:
                        lenergies[method] = {}
                    if method == 'FrequencyIODMD':
                        llti = to_lti(lrom).with_(D=None, sampling_time=1, T=nt, time_stepper=DiscreteTimeStepper())
                        lsols = llti.solve(input=disc_control)
                    else:
                        llti = to_lti(lrom).with_(D=None, T=T, time_stepper=PHDMDImplicitMidpointTimeStepper(nt=nt))
                        lsols = llti.solve(input=control)
                    lenergy = llti.E.pairwise_apply2(lsols, lsols)
                    lenergies[method][order] = lenergy.copy()
                except Exception:
                    pass

                if quad_success:
                    try:
                        if method not in qenergies:
                            qenergies[method] = {}
                            menergies[method] = {}
                        if method == 'FrequencyIODMD':
                            qlti = to_lti(qrom).with_(D=None, sampling_time=1, T=nt, time_stepper=DiscreteTimeStepper())
                            qsols = qlti.solve(input=quad_disc_control)
                            mlti = to_lti(model).with_(D=None, sampling_time=1, T=nt, time_stepper=DiscreteTimeStepper())
                            msols = mlti.solve(input=disc_control)
                        else:
                            qlti = to_lti(qrom).with_(D=None, T=T, time_stepper=PHDMDImplicitMidpointTimeStepper(nt=nt))
                            qsols = qlti.solve(input=quad_control)
                            mlti = to_lti(model).with_(D=None, T=T, time_stepper=PHDMDImplicitMidpointTimeStepper(nt=nt))
                            msols = mlti.solve(input=control)
                        qenergy = qlti.E.pairwise_apply2(qsols, qsols)
                        qenergies[method][order] = qenergy.copy()
                        menergy = mlti.E.pairwise_apply2(msols, msols)
                        menergies[method][order] = menergy.copy()
                    except Exception:
                        pass

        fenergy_df = pd.DataFrame(
            [ ['FOM', '', fom.order] + list(fenergy) ],
            columns=energy_columns
        )
        lenergy_df = pd.DataFrame(columns=energy_columns)
        for method in lenergies:
            nlenergy_df = pd.DataFrame(
                [
                    ['LROM', method, order] + list(lenergies[method][order]) for order in lenergies[method]
                ],
                columns=energy_columns
            )
            lenergy_df = pd.concat([lenergy_df, nlenergy_df], ignore_index=True)
        qenergy_df = pd.DataFrame(columns=energy_columns)
        for method in qenergies:
            nqenergy_df = pd.DataFrame(
                [
                    ['QROM', method, order] + list(qenergies[method][order]) for order in qenergies[method]
                ],
                columns=energy_columns
            )
            qenergy_df = pd.concat([qenergy_df, nqenergy_df], ignore_index=True)
        menergy_df = pd.DataFrame(columns=energy_columns)
        for method in menergies:
            nmenergy_df = pd.DataFrame(
                [
                    ['CROM', method, order] + list(menergies[method][order]) for order in menergies[method]
                ],
                columns=energy_columns
            )
            menergy_df = pd.concat([menergy_df, nmenergy_df], ignore_index=True)

        energy_df = pd.concat([energy_df, fenergy_df, lenergy_df, qenergy_df, menergy_df], ignore_index=True)
        energy_file = os.path.join(FILEPATH, f'{initial}/csv/' + ENERGY.format(initial=initial))
        if not os.path.exists(os.path.dirname(energy_file)):
            os.makedirs(os.path.dirname(energy_file), exist_ok=True)
        energy_df.sort_values(['method', 'ord']).to_csv(energy_file, index=False)

        error_file = os.path.join(FILEPATH, f'{initial}/csv/' + ERROR.format(initial=initial))
        if not os.path.exists(os.path.dirname(error_file)):
            os.makedirs(os.path.dirname(error_file), exist_ok=True)
        error_df.sort_values(['method', 'ord']).to_csv(error_file, index=False)


if __name__ == '__main__':
    run(main)
