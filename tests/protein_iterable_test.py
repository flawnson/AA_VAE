import unittest

import utils.amino_acid_loader as dataloader


class FastaReaderTest(unittest.TestCase):
    def test(self):
        loader = dataloader.ProteinIterableDataset("../data/uniparc_1500_50M", 1500)
        i = 0
        iterator = loader.__iter__()
        for x in iterator:
            i = i + 1
            if (i % 10000) == 0:
                print(i)
        assert True
