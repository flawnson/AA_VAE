import unittest

import utils.data.common as common
import utils.amino_acid_loader as loader


class FlatReaderTest(unittest.TestCase):
    def test(self):
        data = common.read_sequence_from_flat_file("../test_data/test_flat.txt")
        seq, _, _, _ = loader.process_sequences(data, max_length=10, fixed_protein_length=1500, pad_sequence=True)
        assert seq.shape[0] == 10
        assert True

