from quad_ph_mor.experiments.interface import Experiment

from pymordemos.phlti import msd
from pymor.models.iosys import PHLTIModel
from pymor.reductors.ph.ph_irka import PHIRKAReductor


class MSDExperiment(Experiment):
    def __init__(self, log_to_file=True, fix_rng=True):
        super().__init__(name='MSD', log_to_file=log_to_file, fix_rng=fix_rng)

    def setup(self, order):
        self.logger.info(f'Setting up MSD experiment with order {order}...')
        J, R, G, P, S, N, E, Q = msd(n=order, m_i=4., k_i=4., c_i=1.)
        msd_model = PHLTIModel.from_matrices(J, R, G, P, S, N, E, Q).to_berlin_form()
        self.fom_matrices = tuple(msd_model.to_matrices()[:-1])

    def construct_rom(self, order):
        reductor = PHIRKAReductor(self.fom)
        return reductor.reduce(order)
