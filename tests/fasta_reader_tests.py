import unittest

import utils.data.common
import utils.data_load as dataloader
from utils.logger import log


class FastaReaderTest(unittest.TestCase):
    def test(self):
        log.info("Useless info")
        data = utils.data.common.fasta_reader("../test_data/test_fasta.fasta")
        data = [k for k in data]
        print("Useless value")
        assert len(data) == 2
        assert True
        # assert map.keys() == ['protein_id', 'sequence_data']
