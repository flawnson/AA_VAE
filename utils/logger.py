import logging
import os.path as osp
import datetime
import os
import errno


# Callable function to set logger for any module in the repo
def get_file_logger(name, filename, level=logging.DEBUG):
    logger = logging.getLogger(name)
    time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    try:
        os.makedirs(osp.join(osp.dirname(__file__), f"../run_logs"))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e
    log_file = osp.join(osp.dirname(__file__), f"../run_logs", f"{time}-{filename}")
    print(f"Writing logs to {log_file}")
    hdlr = logging.FileHandler(osp.join(osp.dirname(__file__), f"../run_logs", f"{time}-{filename}"))
    formatter = logging.Formatter(fmt='%(asctime)s  %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s ',
                                  datefmt='%Y-%m-%d %I:%M:%S %p')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(level)
    return logger


def get_basic_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logging.basicConfig(format='%(asctime)s  %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s ',
                        datefmt='%Y-%m-%d %I:%M:%S %p',
                        level=level)
    return logger


# log = get_basic_logger(__name__)
log = get_file_logger(__name__, "run.log")
