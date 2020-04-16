import unittest
import collections
import utils.amino_acid_loader as dataloader


class BucketCounterTest(unittest.TestCase):
    def test(self):
        data = [i for i in range(10)]
        data [2] = 3
        data [4] = 3
        counter = collections.Counter(data)
        buckets = dataloader.calculate_bucket_cost(counter, 15, 3)
        data = [k for k in data]
        assert buckets[3] == 1.0
        assert True
        # assert map.keys() == ['protein_id', 'sequence_data']
