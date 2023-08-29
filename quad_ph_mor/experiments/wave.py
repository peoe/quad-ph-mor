import numpy as np

from scipy.sparse import csr_matrix, bmat

from ufl import TestFunction, TrialFunction, dx, ds, grad, inner

from dune.grid import structuredGrid
from dune.fem.operator import galerkin, linear
from dune.fem.space import lagrange
from dune.ufl import Constant, DirichletBC

from pymor.models.iosys import PHLTIModel
from pymor.reductors.ph.ph_irka import PHIRKAReductor

from quad_ph_mor.experiments.interface import Experiment
from quad_ph_mor.tools.misc import get_dune_boundary_dofs


class Wave1DExperiment(Experiment):
    def __init__(self, log_to_file=True, fix_rng=True):
        super().__init__(name='Wave1D', log_to_file=log_to_file, fix_rng=fix_rng)

    def setup(self, order, bounds=(0., 1.), damping=1., wave_speed=1., regularization=1e-8):
        self.logger.info(f'Setting up 1D wave experiment of order {order}')
        mesh = structuredGrid([bounds[0]], [bounds[1]], [order - 1])
        P2 = lagrange(mesh, order=1, dimRange=1)
        P1 = lagrange(mesh, order=1)
        w, v = TrialFunction(P2), TrialFunction(P1)
        phi, psi = TestFunction(P2), TestFunction(P1)

        fom_data = {}

        E_div_form = galerkin([(1. / wave_speed) * inner(w, phi) * dx])
        E_dt_form  = galerkin([inner(v, psi) * dx])
        E_div = linear(E_div_form).as_numpy
        E_dt  = linear(E_dt_form).as_numpy
        fom_data['E'] = bmat([
            [ E_div, None ],
            [ None, E_dt ]
        ]).toarray()

        J_div_form = galerkin([-inner(w, grad(psi)) * dx])
        J_dt_form  = galerkin([inner(grad(v), phi) * dx])
        J_div = linear(J_div_form).as_numpy
        J_dt  = linear(J_dt_form).as_numpy
        fom_data['J'] = bmat([
            [ None, J_dt ],
            [ J_div, None ]
        ]).toarray()

        R_div_form = galerkin([inner(Constant(0.) * w, phi) * dx])
        R_dt_form  = galerkin([damping * inner(v, psi) * dx])
        reg_shape = linear(R_div_form).as_numpy.shape
        R_div = csr_matrix(reg_shape)
        R_div.setdiag(reg_shape[0] * regularization)
        R_dt  = linear(R_dt_form).as_numpy
        fom_data['R'] = bmat([
            [ R_div, None ],
            [ None, R_dt ]
        ]).toarray()

        bc = DirichletBC(P1, 0.)
        G_div_form = galerkin([inner(Constant(0.) * v, psi) * dx, bc]) # bc needed to get boundary dofs!
        bnd_dofs = get_dune_boundary_dofs(G_div_form)
        G_dt_form  = galerkin([inner(v, psi) * ds])
        # G_dt_form  = galerkin([inner(inner(w, normal), psi) * ds])
        G_div = 0 * linear(G_div_form).as_numpy[bnd_dofs].T
        G_dt  = linear(G_dt_form).as_numpy[bnd_dofs].T
        fom_data['G'] = bmat([
            [ G_div ],
            [ G_dt ]
        ]).toarray()

        fom_data['P'] = csr_matrix(fom_data['G'].shape).toarray()

        assert np.allclose(fom_data['J'], -fom_data['J'].T)

        wave_model = PHLTIModel.from_matrices(
            fom_data['J'], fom_data['R'], G=fom_data['G'], P=fom_data['P'], S=None, N=None, E=fom_data['E']
        )
        self.fom_matrices = tuple(wave_model.to_matrices()[:-1])

    def construct_rom(self, order):
        reductor = PHIRKAReductor(self.fom)
        return reductor.reduce(order)
