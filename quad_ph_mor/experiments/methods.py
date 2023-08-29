import numpy as np

from time import time

from pymor.algorithms.phdmd import phdmd
from pymor.algorithms.projection import project
from pymor.algorithms.timestepping import DiscreteTimeStepper, ImplicitEulerTimeStepper
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.exceptions import PHDMDError
from pymor.discretizers.builtin.fv import discretize_instationary_fv
from pymor.discretizers.builtin.grids.oned import OnedGrid
from pymor.models.iosys import LTIModel
from pymor.operators.constructions import NumpyConversionOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace

from quad_ph_mor.tools.iodmd import iodmd, oi
from quad_ph_mor.tools.misc import to_lti, khatri_rao_np, kron_np
from quad_ph_mor.tools.sobmor import sobmor, frequency_sobmor
from quad_ph_mor.tools.sobmor_parameters import SOBMORParameter
from quad_ph_mor.tools.timestepping import PHDMDImplicitMidpointTimeStepper


class Method:
    __NAME__ = ''

    def __init__(self):
        pass

    def _inner(self, **kwargs):
        raise NotImplementedError

    def __call__(self, kwarg_dict):
        return self._inner(**kwarg_dict)


class IODMD(Method):
    __NAME__ = 'IODMD'

    def _inner(self, experiment=None, order=None, control=None, control_expr=None, disc_control=None, nt=100, T=4., rcond=1e-12):
        dt = T / nt
        ts = np.arange(0., T + dt, dt)
        controls = control_expr(ts)

        fom = experiment.fom
        fom_lti = to_lti(fom).with_(T=T, time_stepper=PHDMDImplicitMidpointTimeStepper(nt=nt))
        result = fom_lti.compute(input=control, solution=True, output=True)
        fom_state = result['solution'].to_numpy().T
        fom_output = result['output'].T

        VV = np.linalg.svd(fom_state, full_matrices=False)[0].T
        V = NumpyVectorSpace.from_numpy(VV[:order], id='STATE')

        rom_state = to_matrix(project(NumpyMatrixOperator(fom_state, range_id='STATE'), V, None))
        inf_E = to_matrix(project(fom.E, V, V))

        experiment.logger.debug('Inferring linear model...')

        tic = time()

        inf_model, lin_data = iodmd(X=rom_state, Y=fom_output, U=controls, E=inf_E, rcond=rcond)

        toc = time()

        inf_lti = to_lti(inf_model).with_(sampling_time=1, T=nt, time_stepper=DiscreteTimeStepper())
        inf_lti = inf_lti.with_(D=None)
        inf_output = inf_lti.output(input=disc_control).T
        inf_state = inf_lti.solve(input=disc_control).to_numpy()
        diff_state = khatri_rao_np(inf_state).T

        experiment.logger.debug('Inferring quadratic model...')

        tac = time()

        quad_model, quad_data = iodmd(X=diff_state, Y=(fom_output - inf_output), U=controls, E=kron_np(inf_E), rcond=rcond)

        toe = time()

        quad_lti = to_lti(quad_model)

        result = {
            'order': order,
            'lin_mats': inf_lti.to_matrices(),
            'quad_mats': quad_lti.to_matrices(),
            'lrom_data': lin_data,
            'qrom_data': quad_data,
            'control': control,
            'nt': nt,
            'T': T,
            'rcond': rcond,
            'lin_time': toc - tic,
            'quad_time': toe - tac,
            'lin_success': True,
            'quad_success': True,
            'success': True,
        }

        return result


class FrequencyIODMD(Method):
    __NAME__ = 'FrequencyIODMD'

    def _inner(self, experiment=None, order=None, control=None, control_expr=None, disc_control=None, nt=100, T=4., rcond=1e-12):
        dt = T / nt
        ts = np.arange(0., T + dt, dt)
        controls = control_expr(ts)

        fom = experiment.fom
        fom_lti = to_lti(fom).with_(T=T, time_stepper=PHDMDImplicitMidpointTimeStepper(nt=nt))
        result = fom_lti.compute(input=control, solution=True, output=True)
        fom_state = result['solution'].to_numpy().T
        fom_output = result['output'].T

        VV = np.linalg.svd(fom_state, full_matrices=False)[0].T
        V = NumpyVectorSpace.from_numpy(VV[:order], id='STATE')

        rom_state = to_matrix(project(NumpyMatrixOperator(fom_state, range_id='STATE'), V, None))
        inf_E = to_matrix(project(fom.E, V, V))

        experiment.logger.debug('Inferring linear model...')

        tic = time()

        inf_model, lin_data = iodmd(X=rom_state, Y=fom_output, U=controls, E=inf_E, rcond=rcond)

        toc = time()

        inf_lti = to_lti(inf_model).with_(sampling_time=1, T=nt, time_stepper=DiscreteTimeStepper())
        inf_lti = inf_lti.with_(D=None)
        inf_output = khatri_rao_np(fom_output.T - inf_lti.output(input=disc_control)).T
        inf_state = inf_lti.solve(input=disc_control).to_numpy().T
        diff_state = inf_state

        experiment.logger.debug('Inferring quadratic model...')

        tac = time()

        quad_model, quad_data = iodmd(X=diff_state, Y=inf_output, U=khatri_rao_np(controls.T).T, E=inf_E, rcond=rcond)

        toe = time()

        quad_lti = to_lti(quad_model)

        result = {
            'order': order,
            'lin_mats': inf_lti.to_matrices(),
            'quad_mats': quad_lti.to_matrices(),
            'lrom_data': lin_data,
            'qrom_data': quad_data,
            'control': control,
            'nt': nt,
            'T': T,
            'rcond': rcond,
            'lin_time': toc - tic,
            'quad_time': toe - tac,
            'lin_success': True,
            'quad_success': True,
            'success': True,
        }

        return result


class PHDMD(Method):
    __NAME__ = 'PHDMD'

    def _inner(self, experiment=None, order=None, control=None, control_expr=None, nt=100, T=4., initial_alpha=.1, atol=1e-12, rtol=1e-10, maxiter=5000000):
        dt = T / nt
        ts = np.linspace(0., T, nt + 1)
        controls = control_expr(ts)

        fom = experiment.fom
        fom_lti = to_lti(fom).with_(T=T, time_stepper=PHDMDImplicitMidpointTimeStepper(nt=nt))
        result = fom_lti.compute(input=control, solution=True, output=True)
        fom_state = result['solution'].to_numpy().T
        fom_output = result['output'].T

        VV = np.linalg.svd(fom_state, full_matrices=False)[0].T
        V = NumpyVectorSpace.from_numpy(VV[:order], id='STATE')

        rom_state = to_matrix(project(NumpyMatrixOperator(fom_state, range_id='STATE'), V, None))
        inf_E = to_matrix(project(fom.E, V, V))

        experiment.logger.debug('Inferring linear model...')

        tic = time()

        try:
            inf_model, lin_data = phdmd(X=rom_state, Y=fom_output, U=controls, H=inf_E, dt=dt, maxiter=maxiter, initial_alpha=initial_alpha, atol=atol, rtol=rtol)
            lin_success = True
        except PHDMDError:
            lin_success = False

        toc = time()

        if lin_success:
            inf_lti = to_lti(inf_model).with_(T=T, time_stepper=PHDMDImplicitMidpointTimeStepper(nt=nt))
            inf_lti = inf_lti.with_(D=None)
            inf_output = inf_lti.output(input=control).T
            inf_state = inf_lti.solve(input=control).to_numpy()
            diff_state = khatri_rao_np(inf_state).T

            experiment.logger.debug('Inferring quadratic model...')

            tac = time()

            try:
                quad_model, quad_data = phdmd(X=diff_state, Y=(fom_output - inf_output), U=controls, H=kron_np(inf_E), dt=dt, maxiter=maxiter, initial_alpha=initial_alpha, atol=atol, rtol=rtol, fix_sval_ratio=inf_model.order)
                quad_success = True
            except PHDMDError:
                quad_success = False

            toe = time()
        else:
            quad_success = False

        result = {
            'order': order,
            'lin_mats': to_lti(inf_model).to_matrices() if lin_success else None,
            'quad_mats': to_lti(quad_model).to_matrices() if quad_success else None,
            'lrom_data': lin_data if lin_success else None,
            'qrom_data': quad_data if quad_success else None,
            'control': control,
            'nt': nt,
            'T': T,
            'atol': atol,
            'rtol': rtol,
            'maxiter': maxiter,
            'lin_time': toc - tic,
            'quad_time': toe - tac if quad_success else None,
            'lin_success': lin_success,
            'quad_success': quad_success,
            'success': lin_success and quad_success,
        }

        return result


class FrequencyPHDMD(Method):
    __NAME__ = 'FrequencyPHDMD'

    def _inner(self, experiment=None, order=None, control=None, control_expr=None, nt=100, T=4., initial_alpha=.1, atol=1e-12, rtol=1e-10, maxiter=5000000):
        dt = T / nt
        ts = np.linspace(0., T, nt + 1)
        controls = control_expr(ts)

        fom = experiment.fom
        fom_lti = to_lti(fom).with_(T=T, time_stepper=PHDMDImplicitMidpointTimeStepper(nt=nt))
        result = fom_lti.compute(input=control, solution=True, output=True)
        fom_state = result['solution'].to_numpy().T
        fom_output = result['output'].T

        VV = np.linalg.svd(fom_state, full_matrices=False)[0].T
        V = NumpyVectorSpace.from_numpy(VV[:order], id='STATE')

        rom_state = to_matrix(project(NumpyMatrixOperator(fom_state, range_id='STATE'), V, None))
        inf_E = to_matrix(project(fom.E, V, V))

        experiment.logger.debug('Inferring linear model...')

        tic = time()

        try:
            inf_model, lin_data = phdmd(X=rom_state, Y=fom_output, U=controls, H=inf_E, dt=dt, maxiter=maxiter, initial_alpha=initial_alpha, atol=atol, rtol=rtol)
            lin_success = True
        except PHDMDError:
            lin_success = False

        toc = time()

        if lin_success:
            inf_lti = to_lti(inf_model).with_(T=T, time_stepper=PHDMDImplicitMidpointTimeStepper(nt=nt))
            inf_lti = inf_lti.with_(D=None)
            diff_state = inf_lti.solve(input=control).to_numpy().T
            inf_output = inf_lti.output(input=control).T
            diff_output = khatri_rao_np((fom_output - inf_output).T).T
            diff_controls = khatri_rao_np(controls.T).T

            experiment.logger.debug('Inferring quadratic model...')

            tac = time()

            try:
                quad_model, quad_data = phdmd(X=diff_state, Y=diff_output, U=diff_controls, H=inf_E, dt=dt, maxiter=maxiter, initial_alpha=initial_alpha, atol=atol, rtol=rtol)
                quad_success = True
            except PHDMDError:
                quad_success = False

            toe = time()
        else:
            quad_success = False

        result = {
            'order': order,
            'lin_mats': to_lti(inf_model).to_matrices() if lin_success else None,
            'quad_mats': to_lti(quad_model).to_matrices() if quad_success else None,
            'lrom_data': lin_data if lin_success else None,
            'qrom_data': quad_data if quad_success else None,
            'control': control,
            'nt': nt,
            'T': T,
            'atol': atol,
            'rtol': rtol,
            'maxiter': maxiter,
            'lin_time': toc - tic,
            'quad_time': toe - tac if quad_success else None,
            'lin_success': lin_success,
            'quad_success': quad_success,
            'success': lin_success and quad_success,
        }

        return result


class OI(Method):
    __NAME__ = 'OI'

    def _inner(self, experiment=None, order=None, control=None, control_expr=None, nt=100, T=4.):
        dt = T / nt
        ts = np.linspace(0., T, nt + 1)
        controls = control_expr(ts)

        fom = experiment.fom
        fom_lti = to_lti(fom).with_(T=T, time_stepper=PHDMDImplicitMidpointTimeStepper(nt=nt))
        result = fom_lti.compute(input=control, solution=True, output=True)
        fom_state = result['solution'].to_numpy().T
        fom_output = result['output'].T

        VV = np.linalg.svd(fom_state, full_matrices=False)[0].T
        V = NumpyVectorSpace.from_numpy(VV[:order], id='STATE')

        rom_state = to_matrix(project(NumpyMatrixOperator(fom_state, range_id='STATE'), V, None))
        inf_E = to_matrix(project(fom.E, V, V))

        experiment.logger.debug('Inferring linear model...')

        tic = time()

        inf_model, lin_data = oi(X=rom_state, Y=fom_output, U=controls, E=inf_E, dt=dt)

        toc = time()

        inf_lti = to_lti(inf_model).with_(T=T, time_stepper=PHDMDImplicitMidpointTimeStepper(nt=nt))
        inf_lti = inf_lti.with_(D=None)
        inf_output = inf_lti.output(input=control).T
        inf_state = inf_lti.solve(input=control).to_numpy()
        diff_state = khatri_rao_np(inf_state).T

        experiment.logger.debug('Inferring quadratic model...')

        tac = time()

        quad_model, quad_data = oi(X=diff_state, Y=(fom_output - inf_output), U=controls, E=kron_np(inf_E), dt=dt)

        toe = time()

        result = {
            'order': order,
            'lin_mats': to_lti(inf_model).to_matrices(),
            'quad_mats': to_lti(quad_model).to_matrices(),
            'lrom_data': lin_data,
            'qrom_data': quad_data,
            'control': control,
            'nt': nt,
            'T': T,
            'lin_time': toc - tic,
            'quad_time': toe - tac,
            'lin_success': True,
            'quad_success': True,
            'success': True,
        }

        return result


class FrequencyOI(Method):
    __NAME__ = 'FrequencyOI'

    def _inner(self, experiment=None, order=None, control=None, control_expr=None, nt=100, T=4.):
        dt = T / nt
        ts = np.linspace(0., T, nt + 1)
        controls = control_expr(ts)

        fom = experiment.fom
        fom_lti = to_lti(fom).with_(T=T, time_stepper=PHDMDImplicitMidpointTimeStepper(nt=nt))
        result = fom_lti.compute(input=control, solution=True, output=True)
        fom_state = result['solution'].to_numpy().T
        fom_output = result['output'].T

        VV = np.linalg.svd(fom_state, full_matrices=False)[0].T
        V = NumpyVectorSpace.from_numpy(VV[:order], id='STATE')

        rom_state = to_matrix(project(NumpyMatrixOperator(fom_state, range_id='STATE'), V, None))
        inf_E = to_matrix(project(fom.E, V, V))

        experiment.logger.debug('Inferring linear model...')

        tic = time()

        inf_model, lin_data = oi(X=rom_state, Y=fom_output, U=controls, E=inf_E, dt=dt)

        toc = time()

        inf_lti = to_lti(inf_model).with_(T=T, time_stepper=PHDMDImplicitMidpointTimeStepper(nt=nt))
        inf_lti = inf_lti.with_(D=None)
        inf_output = khatri_rao_np(fom_output.T - inf_lti.output(input=control)).T
        inf_state = inf_lti.solve(input=control).to_numpy().T
        diff_state = inf_state

        experiment.logger.debug('Inferring quadratic model...')

        tac = time()

        quad_model, quad_data = oi(X=diff_state, Y=inf_output, U=khatri_rao_np(controls.T).T, E=inf_E, dt=dt)

        toe = time()

        result = {
            'order': order,
            'lin_mats': to_lti(inf_model).to_matrices(),
            'quad_mats': to_lti(quad_model).to_matrices(),
            'lrom_data': lin_data,
            'qrom_data': quad_data,
            'control': control,
            'nt': nt,
            'T': T,
            'lin_time': toc - tic,
            'quad_time': toe - tac,
            'lin_success': True,
            'quad_success': True,
            'success': True,
        }

        return result
