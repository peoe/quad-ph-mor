import numpy as np

from pymor.algorithms.timestepping import TimeStepper, _depends_on_time
from pymor.operators.interface import Operator
from pymor.parameters.base import Mu
from pymor.vectorarrays.interface import VectorArray

from quad_ph_mor.tools.misc import khatri_rao_np


class PHDMDImplicitMidpointTimeStepper(TimeStepper):
    """Implicit midpoint rule time-stepper. Symplectic integrator + preserves quadratic invariants.

    Solves equations of the form ::

        M(mu) * d_t u + A(u, mu, t) = F(mu, t),
                         u(mu, t_0) = u_0(mu).

    by implicit midpoint time integration.

    Parameters
    ----------
    nt
        The number of time-steps the time-stepper will perform.
    solver_options
        The |solver_options| used to invert `M - dt/2*A`.
        The special values `'mass'` and `'operator'` are
        recognized, in which case the solver_options of
        M (resp. A) are used.
    """

    def __init__(self, nt, solver_options='operator'):
        self.__auto_init(locals())

    def estimate_time_step_count(self, initial_time, end_time):
        return self.nt

    def iterate(self, initial_time, end_time, initial_data, operator, rhs=None, mass=None, mu=None, num_values=None):
        if not operator.linear:
            raise NotImplementedError
        A, F, E, U0, t0, t1, nt = operator, rhs, mass, initial_data, initial_time, end_time, self.nt
        assert isinstance(A, Operator)
        assert isinstance(F, (type(None), Operator, VectorArray))
        assert isinstance(E, (type(None), Operator))
        assert A.source == A.range
        num_values = num_values or nt + 1
        dt = (t1 - t0) / nt
        DT = (t1 - t0) / (num_values - 1)

        if F is None:
            F_time_dep = False
        elif isinstance(F, Operator):
            assert F.source.dim == 1
            assert F.range == A.range
            F_time_dep = _depends_on_time(F, mu)
            if not F_time_dep:
                dt_F = F.as_vector(mu) * dt
        else:
            assert len(F) == 1
            assert F in A.range
            F_time_dep = False
            dt_F = F * dt

        if E is None:
            from pymor.operators.constructions import IdentityOperator
            M = IdentityOperator(A.source)

        assert A.source == E.source == E.range
        assert not E.parametric
        assert U0 in A.source
        assert len(U0) == 1

        num_ret_values = 1
        yield U0, t0

        if self.solver_options == 'operator':
            options = A.solver_options
        elif self.solver_options == 'mass':
            options = E.solver_options
        else:
            options = self.solver_options

        E_dt_A_impl = (E + A * (dt/2)).with_(solver_options=options)
        if not _depends_on_time(E_dt_A_impl, mu):
            E_dt_A_impl = E_dt_A_impl.assemble(mu)
        E_dt_A_expl = (E - A * (dt/2)).with_(solver_options=options)
        if not _depends_on_time(E_dt_A_expl, mu):
            E_dt_A_expl = E_dt_A_expl.assemble(mu)

        t = t0
        U = U0.copy()
        if mu is None:
            mu = Mu()

        for n in range(nt):
            mu1 = mu.with_(t=t).to_numpy()
            mu2 = mu.with_(t=t + dt).to_numpy()
            mu_helper = mu.parameters.parse(.5 * (mu1 + mu2))
            t += dt
            rhs = E_dt_A_expl.apply(U)
            if F_time_dep:
                dt_F = F.as_vector(mu_helper) * dt
            if F:
                rhs += dt_F

            U = E_dt_A_impl.apply_inverse(rhs)

            while t - t0 + (min(dt, DT) * 0.5) >= num_ret_values * DT:
                num_ret_values += 1
                yield U, t
