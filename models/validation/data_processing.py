import torch
import numpy as np
import pandas as pd
import os.path as osp

from torch.utils.data.dataset import Dataset
from abc import ABCMeta, abstractmethod, ABC
from sklearn.preprocessing import OneHotEncoder


class EmbeddingData(Dataset, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, embedding_dict, onehot):
        """
        Factory class to create datasets that consist of protein embedding and label pairs.
        self.k represents the known mask (in a semi-supervised task this would be useful,
        as is implemented in the GAT model. For this linear model however, it is used to
        ensure any unknown protein embeddings are not learnt by the model (the linear will
        be trained in a supervised manner)

        :param embeddings: torch tensor of embeddings
        :param targets: torch tensor of corresponding targets
        """
        self.onehot = onehot
        self.x = embedding_dict
        if self.onehot:
            self.y = [item[1] for item in self.onehot_encoder()]
        else:
            self.y = [item[1] for item in self.integer_encoder()]

    def label_mapper(self):
        return {}

    def onehot_encoder(self):
        embed_dict: dict = dict(zip(self.x["gene"].values(), self.x["embeddings"].values()))
        label_dict: dict = self.label_mapper()

        intersect = embed_dict.keys() & label_dict.keys()
        filter_embed = {key: np.squeeze(value) for key, value in embed_dict.items() if key in intersect}  # Extra dim

        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = np.asarray(list(label_dict.values())).reshape(len(list(label_dict.values())), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        filter_label = dict(zip(list(label_dict.keys()), onehot_encoded))
        pairs = [[filter_embed[gene], filter_label[gene]] for gene in intersect]

        return pairs

    def integer_encoder(self) -> list:
        embed_dict: dict = dict(zip(self.x["gene"].values(), self.x["embeddings"].values()))
        label_dict: dict = self.label_mapper()

        intersect = embed_dict.keys() & label_dict.keys()
        filter_embed = {key: np.squeeze(value) for key, value in embed_dict.items() if key in intersect}  # Extra dim
        filter_label = {key: value for key, value in label_dict.items() if key in intersect}

        pairs = [[filter_embed[gene], filter_label[gene]] for gene in intersect]

        return pairs

    def __getitem__(self, idx):
        if self.onehot:
            examples = self.onehot_encoder()
        else:
            examples = self.integer_encoder()
        return examples[idx][0], examples[idx][1]

    def __len__(self):
        return len(self.integer_encoder())


class BinaryLabels(EmbeddingData, ABC):
    def __init__(self, embedding_dict, onehot):
        super(BinaryLabels, self).__init__(embedding_dict=embedding_dict, onehot=onehot)

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

        :return: Dictionary of labels that can be used to map the data labels to the classes.
        """
        node_labels = pd.read_csv(osp.join(osp.dirname(__file__), "raw_data", "binary_labels.csv"), header=0)
        data = []
        for symbol in ["hgnc_symbol", "PosHit", "NegHit"]:
            data.append(node_labels[symbol].tolist())

        return {name: self.get_label(positive, negative) for name, positive, negative in
                zip(data[0], data[1], data[2])}


class QuaternaryLabels(EmbeddingData, ABC):
    def __init__(self, embedding_dict, onehot):
        super(QuaternaryLabels, self).__init__(embedding_dict=embedding_dict, onehot=onehot)

    def get_label(self, query: str):
        """

        :param query: a gene name that is requesting a label
        :return: Numerical label
        """

        mappings: dict = {
            'fibrous_proteins': 1,
            'membrane_proteins': 2,
            'unstructured_proteins': 3,
            'globular_proteins': 4
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
    def __init__(self, embedding_dict, onehot):
        super(QuinaryLabels, self).__init__(embedding_dict=embedding_dict, onehot=onehot)

    def get_label(self, query: str):
        """

        :param query: a gene name that is requesting a label
        :return: Numerical label
        """
        mappings: dict = {
            1000000: 1,
            1000001: 2,
            1000002: 3,
            1000003: 4,
            1000004: 5
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
    def __init__(self, embedding_dict, onehot):
        super(ProteinLabels, self).__init__(embedding_dict=embedding_dict, onehot=onehot)

    def data_file(self):
        """
        Due to how the get_label method functions, the file needs to be opened within the child class

        :return: Numerical label
        """
        node_labels = pd.read_csv(osp.join(osp.dirname(__file__), "raw_data", "protein_labels.csv"), header=0)
        data = []
        for symbol in ["gene", "label"]:
            data.append(node_labels[symbol].tolist())

        return data

    def get_label(self, query: str):
        """

        :param query: a gene name that is requesting a label (protein labels are numerous, hence a range(len) is used
        :return: Numerical label
        """
        mappings = dict(zip(np.unique(self.data_file()[1]), list(range(1, len(np.unique(self.data_file()[1]))))))
        if mappings.keys().__contains__(query):
            return mappings.get(query)
        else:
            # BUG: Problem with sklearn's one hot encoder, will return an index error if in case of value without label
            #      This is due to the fact that sklearn starts its indices at 0
            print("There is a value that cannot be converted to a label")
            return 0

    def label_mapper(self):
        return {name: self.get_label(label) for name, label in zip(self.data_file()[0], self.data_file()[1])}

