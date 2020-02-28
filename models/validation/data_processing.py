import numpy as np
import pandas as pd
import os.path as osp

from torch.utils.data.dataset import Dataset
from abc import ABCMeta, abstractmethod, ABC
from sklearn.preprocessing import OneHotEncoder


class EmbeddingData(Dataset, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, embedding_dict, onehot=False):
        """

        :param embeddings: torch tensor of embeddings
        :param targets: torch tensor of corresponding targets
        """
        self.x = embedding_dict
        self.onehot = onehot

    def label_mapper(self):
        return {}

    def onehot_encoder(self, label_dict):
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = np.asarray(list(label_dict.values())).reshape(len(list(label_dict.values())), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

        return onehot_encoded

    def preprocessing(self) -> list:
        embed_dict: dict = dict(zip(self.x["gene"].values(), self.x["embeddings"].values()))
        label_dict: dict = self.label_mapper()

        intersect = embed_dict.keys() & label_dict.keys()
        filter_embed = {key: np.squeeze(value) for key, value in embed_dict.items() if key in intersect}  # Extra dim
        filter_label = {key: value for key, value in label_dict.items() if key in intersect}

        if self.onehot:
            onehot_labels = self.onehot_encoder(filter_label)
            filter_label = dict(zip(list(filter_label.keys()), onehot_labels))

        pairs = [[filter_embed[gene], filter_label[gene]] for gene in intersect]

        return pairs

    def __getitem__(self, idx):
        examples = self.preprocessing()
        return examples[idx][0], examples[idx][1]

    def __len__(self):
        return len(self.preprocessing())


class BinaryLabels(EmbeddingData, ABC):
    def __init__(self, embedding_dict):
        super(BinaryLabels, self).__init__(embedding_dict=embedding_dict)

    def get_label(self, positive, negative):
        """

        :param positive: Positive hit data as determined by the input label file
        :param negative: Negative hit data as determined by the input label file
        :return: Numerical label
        """
        if positive > 0:
            return 1
        elif negative > 0:
            return 2
        elif positive > 0 and negative > 0:
            return 2
        elif positive == 0 and negative == 0:
            return 0
        else:
            print("There is a value that cannot be converted to a label")
            return 0

    def label_mapper(self):
        """

        :param data: HGNC symbol
        :return: Dictionary of labels that can be used to map the data labels to the classes.
        """
        node_labels = pd.read_csv(osp.join(osp.dirname(__file__), "raw_data", "binary_labels.csv"), header=0)
        data = []
        for symbol in ["hgnc_symbol", "PosHit", "NegHit"]:
            data.append(node_labels[symbol].tolist())

        return {name: self.get_label(positive, negative) for name, positive, negative in
                zip(data[0], data[1], data[2])}


class QuaternaryLabels(EmbeddingData, ABC):
    def __init__(self, embedding_dict):
        super(QuaternaryLabels, self).__init__(embedding_dict=embedding_dict)

    def get_label(self, query: str):
        mappings: dict = {
            'fibrous_proteins': 0,
            'membrane_proteins': 1,
            'unstructured_proteins': 2,
            'globular_proteins': 3
        }
        if mappings.keys().__contains__(query):
            return mappings.get(query)
        else:
            print("There is a value that cannot be converted to a label")
            return 0

    def label_mapper(self):
        node_labels = pd.read_csv(osp.join(osp.dirname(__file__), "raw_data", "quaternary_labels.csv"), header=0)
        data = []
        for symbol in ["hgnc_symbol", "protein_type"]:
            data.append(node_labels[symbol].tolist())

        return {name: self.get_label(label) for name, label in zip(data[0], data[1])}


class QuinaryLabels(EmbeddingData, ABC):
    def __init__(self, embedding_dict):
        super(QuinaryLabels, self).__init__(embedding_dict=embedding_dict)

    def get_label(self, query: str):
        mappings: dict = {
            1000000: 0,
            1000001: 1,
            1000002: 2,
            1000003: 3,
            1000004: 4
        }

        if mappings.keys().__contains__(query):
            return mappings.get(query)
        else:
            print("There is a value that cannot be converted to a label")
            return 0

    def label_mapper(self):
        node_labels = pd.read_csv(osp.join(osp.dirname(__file__), "raw_data", "quinary_labels.csv"), header=0)
        data = []
        for symbol in ["gene", "label"]:
            data.append(node_labels[symbol].tolist())

        return {name: self.get_label(label) for name, label in zip(data[0], data[1])}


class ProteinLabels(EmbeddingData, ABC):
    def __init__(self, embedding_dict):
        super(ProteinLabels, self).__init__(embedding_dict=embedding_dict)

    def data_file(self):
        node_labels = pd.read_csv(osp.join(osp.dirname(__file__), "raw_data", "protein_labels.csv"), header=0)
        data = []
        for symbol in ["gene", "label"]:
            data.append(node_labels[symbol].tolist())

        return data

    def get_label(self, query: str):
        mappings = dict(zip(np.unique(self.data_file()[1]), list(range(0, len(np.unique(self.data_file()[1]))))))
        if mappings.keys().__contains__(query):
            return mappings.get(query)
        else:
            # BUG: Problem with sklearn's one hot encoder, will return an index error if in case of value without label
            #      This is due to the fact that sklearn starts its indices at 0
            print("There is a value that cannot be converted to a label")
            return 0

    def label_mapper(self):
        return {name: self.get_label(label) for name, label in zip(self.data_file()[0], self.data_file()[1])}

