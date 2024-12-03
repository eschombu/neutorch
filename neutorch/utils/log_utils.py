import logging
import os
import platform
import sys
from datetime import datetime
from typing import Optional

_neutorch_logger: Optional[logging.Logger] = None
_format_parts = [
    '%(asctime)s',
    '%(name)s',
    'Process %(process)d',
    '%(filename)s',
    'Line %(lineno)d',
    '%(levelname)s',
    '%(message)s',
]


def _get_path() -> str:
    slurm_vars = ['SLURM_JOB_ID', 'SLURM_PROCID']
    global_parts = ['neutorch']
    job_parts = []
    for var in slurm_vars:
        if os.environ.get(var):
            job_parts.append(os.environ[var])
    if len(job_parts) == 0:
        job_parts.append(platform.node())
        job_parts.append(datetime.now().strftime('%Y%m%dT%H%M%S'))
    return '_'.join(global_parts + job_parts) + '.log'


def _get_formatter(fmt=(' - '.join(_format_parts))) -> logging.Formatter:
    return logging.Formatter(fmt)


def _init_logger(path=None, stdout=True, stdout_only=False, formatter=_get_formatter(), level=logging.INFO):
    global _neutorch_logger
    if _neutorch_logger is None:
        _neutorch_logger = logging.getLogger('neutorch')
        if isinstance(formatter, str):
            formatter = _get_formatter(formatter)
        if not stdout_only:
            if path is None:
                path = _get_path()
            filehandler = logging.FileHandler(path)
            filehandler.setFormatter(formatter)
            _neutorch_logger.addHandler(filehandler)
        if stdout:
            streamhandler = logging.StreamHandler(sys.stdout)
            streamhandler.setFormatter(formatter)
            _neutorch_logger.addHandler(streamhandler)
        _neutorch_logger.setLevel(level)


def get_logger() -> logging.Logger:
    global _neutorch_logger
    if _neutorch_logger is None:
        _init_logger()
    return _neutorch_logger


def set_level(level: int):
    global _neutorch_logger
    if _neutorch_logger is None:
        _init_logger()
    _neutorch_logger.setLevel(level)