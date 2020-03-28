import logging

log = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s  %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s ',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)
