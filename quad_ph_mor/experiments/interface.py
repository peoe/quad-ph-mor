import numpy as np

from datetime import datetime
from time import sleep

from pymor.models.iosys import PHLTIModel

from quad_ph_mor.tools.data import save
from quad_ph_mor.tools.logging import ExperimentLogger
from quad_ph_mor.tools.misc import to_lti


class Experiment:
    DATA_BASE_NAME = '{date}_{name}'

    fom_matrices = None
    solvable_fom = None
    reductor_class = None
    rom_matrices = {}
    rom_lti_matrices = {}

    def __init__(self, name, log_to_file=True, fix_rng=True):
        self.name = name
        self.logger = ExperimentLogger(name, file=log_to_file)

        if fix_rng:
            np.random.seed(42)

    @property
    def fom(self):
        return PHLTIModel.from_matrices(*tuple(self.fom_matrices))

    @property
    def fom_lti(self):
        return to_lti(self.fom)

    def roms(self, order):
        return PHLTIModel.from_matrices(*tuple(self.rom_matrices[order]))

    def rom_ltis(self, order):
        return to_lti(self.roms(order))

    def setup(self):
        raise NotImplementedError

    def construct_rom(self, order):
        raise NotImplementedError

    def reduce(self, order, overwrite=False):
        if order in self.rom_matrices and not overwrite:
            self.logger.warn(f'ROM {order} already exists, ignoring. Set overwrite to True if you want to force!')
        else:
            if overwrite:
                self.logger.debug(f'Overriding ROM of order {order}...')
            self.logger.info(f'Reducing to order {order}...')
            rom = self.construct_rom(order)
            rom_lti = to_lti(rom)
            self.rom_matrices[order] = rom.to_matrices()[:-1]
            self.rom_lti_matrices[order] = rom_lti.to_matrices()[:-1]

    def save_data(self, initial, **data):
        # safety delay because we can save only every other second
        # otherwise, useful data will be overwritten...
        sleep(1.1)
        name = self.DATA_BASE_NAME.format(date=f'{datetime.now():%Y_%m_%d_%H_%M_%S}', name=self.name)

        self.logger.debug(f'Saving data under {name}...')
        save(f'{self.name.lower()}/{initial}/' + name, data=data)
