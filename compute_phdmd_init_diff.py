import os

import numpy as np
import pandas as pd

from typer import run
from pkg_resources import resource_filename

from pymor.algorithms.phdmd import phdmd, _weighted_phdmd
from pymor.algorithms.projection import project
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.exceptions import PHDMDError
from pymor.core.logger import getLogger
from pymor.models.iosys import LTIModel, PHLTIModel
from pymor.operators.block import BlockColumnOperator, BlockDiagonalOperator, BlockRowOperator
from pymor.operators.constructions import ComponentProjectionOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

from quad_ph_mor.experiments.msd import MSDExperiment
from quad_ph_mor.tools.timestepping import PHDMDImplicitMidpointTimeStepper
from quad_ph_mor.tools.operator import QuadraticInputOperator
from quad_ph_mor.tools.misc import to_lti, khatri_rao_np


FILEPATH = resource_filename('quad_ph_mor', 'experiments/data/msd/phdmd_init_comp')


def main():
    logger = getLogger('pymor')
    logger.setLevel('ERROR')

    msd = MSDExperiment(log_to_file=True)
    msd.setup(order=100)
    fom = msd.fom
    fom_h2 = fom.h2_norm()
    fom_hinf = fom.hinf_norm()
    red_orders = range(2, 11)
    control = '[(t[0] < .5) * 1., (t[0] < .5) * -1.]'
    control_expr = lambda t: np.vstack([np.where(t < .5, 1., 0.), np.where(t < .5, -1., 0.)])
    T=4.
    nt = 10000
    dt = T / nt
    ts = np.linspace(0., T, nt + 1)
    controls = control_expr(ts)

    msd.logger.info('Running init comp MSD...')

    columns = ['method', 'ord', 'FOMh2', 'ilh2', 'rilh2', 'lh2', 'rlh2', 'iqh2', 'riqh2', 'qh2', 'rqh2', 'FOMhinf', 'ilhinf', 'rilhinf', 'lhinf', 'rlhinf', 'iqhinf', 'riqhinf', 'qhinf', 'rqhinf']
    df = pd.DataFrame(columns=columns)

    for n_meth, order in enumerate(red_orders):
        msd.logger.debug(f'Comparing order {n_meth}...')

        fom = msd.fom
        fom_lti = to_lti(fom).with_(T=T, time_stepper=PHDMDImplicitMidpointTimeStepper(nt=nt))
        result = fom_lti.compute(input=control, solution=True, output=True)
        fom_state = result['solution'].to_numpy().T
        fom_output = result['output'].T

        VV = np.linalg.svd(fom_state, full_matrices=False)[0].T
        V = NumpyVectorSpace.from_numpy(VV[:order], id='STATE')

        rom_state = to_matrix(project(NumpyMatrixOperator(fom_state, range_id='STATE'), V, None))
        inf_E = to_matrix(project(fom.E, V, V))

        Xdot = (1. / dt) * (rom_state[:, 1:] - rom_state[:, :-1])
        X = .5 * (rom_state[:, 1:] + rom_state[:, :-1])
        Y = .5 * (fom_output[:, 1:] + fom_output[:, :-1])
        U = .5 * (controls[:, 1:] + controls[:, :-1])

        state_dim = X.shape[0]

        msd.logger.debug('Computing initial linear model...')

        lT = np.concatenate([X, U])
        lZ = np.concatenate([inf_E @ Xdot, -Y])
        initial_J, initial_R, weighted_data = _weighted_phdmd(lT, lZ)
        init_J = initial_J[:state_dim, :state_dim]
        init_G = initial_J[:state_dim, state_dim:]
        init_N = initial_J[state_dim:, state_dim:]
        init_R = initial_R[:state_dim, :state_dim]
        init_P = initial_R[:state_dim, state_dim:]
        init_S = initial_R[state_dim:, state_dim:]
        linit_model = PHLTIModel.from_matrices(
            J=init_J, R=init_R, G=init_G, P=init_P, N=init_N, S=init_S, E=inf_E
        )

        msd.logger.debug('Inferring linear model...')

        try:
            inf_model, lin_data = phdmd(X=rom_state, Y=fom_output, U=controls, H=inf_E, dt=dt, maxiter=5000000)
            lin_success = True
        except PHDMDError:
            lin_success = False

        if lin_success:
            inf_lti = to_lti(inf_model).with_(T=T, time_stepper=PHDMDImplicitMidpointTimeStepper(nt=nt))
            inf_lti = inf_lti.with_(D=None)
            diff_state = inf_lti.solve(input=control).to_numpy().T
            inf_output = inf_lti.output(input=control).T
            diff_output = khatri_rao_np((fom_output - inf_output).T).T
            diff_controls = khatri_rao_np(controls.T).T

            msd.logger.debug('Inferring quadratic model...')

            try:
                quad_model, quad_data = phdmd(X=diff_state, Y=diff_output, U=diff_controls, H=inf_E, dt=dt, maxiter=5000000)
                quad_success = True

                B = BlockRowOperator([ inf_model.B, quad_model.B ])
                C = BlockColumnOperator([ inf_model.C, quad_model.C ])
                D = BlockDiagonalOperator([ inf_model.D, quad_model.D ])
                q_op = QuadraticInputOperator(inf_model.B.source)
                c_op = ComponentProjectionOperator(range(inf_model.B.source.dim), C.range)
                B = B @ q_op
                C = c_op @ C
                D = c_op @ D @ q_op

                A = inf_model.A + quad_model.A
                E = inf_model.E + quad_model.E

                model = LTIModel(A=A, B=B, C=C, E=E, D=D)
            except PHDMDError:
                quad_success = False

            qXdot = (1. / dt) * (diff_state[:, 1:] - diff_state[:, :-1])
            qX = .5 * (diff_state[:, 1:] + diff_state[:, :-1])
            qY = .5 * (diff_output[:, 1:] + diff_output[:, :-1])
            qU = .5 * (diff_controls[:, 1:] + diff_controls[:, :-1])

            state_dim = qX.shape[0]

            msd.logger.debug('Computing initial quadratic model...')

            qT = np.concatenate([qX, qU])
            qZ = np.concatenate([inf_E @ qXdot, -qY])
            initial_J, initial_R, weighted_data = _weighted_phdmd(qT, qZ)
            init_J = initial_J[:state_dim, :state_dim]
            init_G = initial_J[:state_dim, state_dim:]
            init_N = initial_J[state_dim:, state_dim:]
            init_R = initial_R[:state_dim, :state_dim]
            init_P = initial_R[:state_dim, state_dim:]
            init_S = initial_R[state_dim:, state_dim:]
            qinit_model = PHLTIModel.from_matrices(
                J=init_J, R=init_R, G=init_G, P=init_P, N=init_N, S=init_S, E=inf_E
            )

            B = BlockRowOperator([ inf_model.B, qinit_model.B ])
            C = BlockColumnOperator([ inf_model.C, qinit_model.C ])
            D = BlockDiagonalOperator([ inf_model.D, qinit_model.D ])
            q_op = QuadraticInputOperator(inf_model.B.source)
            c_op = ComponentProjectionOperator(range(inf_model.B.source.dim), C.range)
            B = B @ q_op
            C = c_op @ C
            D = c_op @ D @ q_op

            A = inf_model.A + qinit_model.A
            E = inf_model.E + qinit_model.E

            qinit_model = LTIModel(A=A, B=B, C=C, E=E, D=D)

        ndf = pd.DataFrame([[
                'FrequencyPHDMD',
                order,
                fom_h2,
                (fom - linit_model).h2_norm() if lin_success else None,
                (fom - linit_model).h2_norm() / fom_h2 if lin_success else None,
                (fom - inf_model).h2_norm() if lin_success else None,
                (fom - inf_model).h2_norm() / fom_h2 if lin_success else None,
                (fom - qinit_model).h2_norm() if quad_success else None,
                (fom - qinit_model).h2_norm() / fom_h2 if quad_success else None,
                (fom - model).h2_norm() if quad_success else None,
                (fom - model).h2_norm() / fom_h2 if quad_success else None,
                fom_hinf,
                (fom - linit_model).hinf_norm() if lin_success else None,
                (fom - linit_model).hinf_norm() / fom_hinf if lin_success else None,
                (fom - inf_model).hinf_norm() if lin_success else None,
                (fom - inf_model).hinf_norm() / fom_hinf if lin_success else None,
                (fom - qinit_model).hinf_norm() if quad_success else None,
                (fom - qinit_model).hinf_norm() / fom_hinf if quad_success else None,
                (fom - model).hinf_norm() if quad_success else None,
                (fom - model).hinf_norm() / fom_hinf if quad_success else None,
            ]],
            columns=columns
        )
        df = pd.concat([df, ndf], ignore_index=True)

    df = df.replace([np.inf, -np.inf], np.nan)
    init_file = os.path.join(FILEPATH, 'init_diff.csv')
    if not os.path.exists(os.path.dirname(init_file)):
        os.makedirs(os.path.dirname(init_file), exist_ok=True)
    df.to_csv(init_file, index=False)


if __name__ == '__main__':
    run(main)
