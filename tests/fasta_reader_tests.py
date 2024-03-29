import unittest

import utils.data.common
import utils.amino_acid_loader as dataloader
from utils.logger import log


class FastaReaderTest(unittest.TestCase):
    def test(self):
        data = utils.data.common.fasta_reader("../test_data/test_fasta.fasta")
        data = [k for k in data]
        assert len(data) == 2
        assert True
        # assert map.keys() == ['protein_id', 'sequence_data']
