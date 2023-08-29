import logging
import os

from pkg_resources import resource_filename


FILEPATH = resource_filename('quad_ph_mor', 'experiments/data')


class ExperimentLogger:
    def __init__(self, name, file=True):
        self.name = name
        self._logger = logging.getLogger(self.name)
        formatter = logging.Formatter('%(asctime)s [%(levelname)-5.5s] %(message)s')

        self._logger.setLevel(logging.INFO)
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        self._logger.addHandler(streamHandler)

        if file:
            self._file_logger = logging.getLogger(self.name + '_file')
            self._file_logger.setLevel(logging.DEBUG)
            log_file = os.path.join(FILEPATH, f'{name.lower()}/' + name + '.log')
            if not os.path.exists(os.path.dirname(log_file)):
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fileHandler = logging.FileHandler(log_file)
            fileHandler.setFormatter(formatter)
            self._file_logger.addHandler(fileHandler)
        else:
            self._file_logger = None

    def warn(self, msg):
        self._logger.warn(msg)
        if self._file_logger is not None:
            self._file_logger.info(msg)

    def info(self, msg):
        self._logger.info(msg)
        if self._file_logger is not None:
            self._file_logger.info(msg)

    def debug(self, msg):
        if self._file_logger is not None:
            self._file_logger.debug(msg)
