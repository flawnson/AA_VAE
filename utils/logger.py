import datetime
import errno
import logging
import os
import os.path as osp

log_format = '[%(levelname)-8s %(asctime)s %(filename)s:%(lineno)d] %(message)s '
datefmt = '%Y-%m-%d %I:%M:%S %p'


# Callable function to set logger for any module in the repo
def set_file_logger(name='', filename="run.log", level=logging.DEBUG):
    logger = logging.getLogger(name)
    logging.basicConfig(format=log_format,
                        datefmt=datefmt,
                        level=level)
    time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    try:
        os.makedirs(osp.join(osp.dirname(__file__), f"../run_logs"))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e
    log_file = osp.join(osp.dirname(__file__), f"../run_logs", f"{time}-{filename}")
    print(f"Writing logs to {log_file}")
    hdlr = logging.FileHandler(osp.join(osp.dirname(__file__), f"../run_logs", f"{time}-{filename}"))
    log_formatter = logging.Formatter(fmt=log_format,
                                      datefmt=datefmt)
    hdlr.setFormatter(log_formatter)
    logger.addHandler(hdlr)
    logger.setLevel(level)
    return logger


def get_logger(name):
    logger = logging.getLogger(name)
    return logger


console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter(log_format, datefmt=datefmt)
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

log = get_logger(__name__)
