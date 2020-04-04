import unittest

import utils.data as dataloader
from utils.logger import log


class FastaReaderTest(unittest.TestCase):
    def test(self):
        log.info("Useless info")
        data = dataloader.fasta_reader("../test_data/test_fasta.fasta")
        data = [k for k in data]
        assert len(data) == 2
        assert True
        # assert map.keys() == ['protein_id', 'sequence_data']
