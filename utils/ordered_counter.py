import collections
import unittest


class OrderedCounterBase(collections.Counter, collections.OrderedDict):
    """Counter that remembers the order elements are first encountered"""

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, collections.OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (collections.OrderedDict(self),)


class OrderedCounter(OrderedCounterBase):
    def __init__(self, datalist):
        super().__init__(sorted(datalist))


class OrderedCounterTest(unittest.TestCase):
    def test(self):
        datalist = OrderedCounter([4, 5, 6])
        for k, v in datalist.items():
            print(f"{k} {v}")
