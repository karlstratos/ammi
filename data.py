import numpy as np
import pickle
import scipy.io
import torch

from torch.utils.data import Dataset, DataLoader, TensorDataset


class Data:

    def __init__(self, file_path):
        self.file_path = file_path
        self.load_datasets()

    def get_loaders(self, batch_size, num_workers, shuffle_train=False,
                    get_test=True):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=shuffle_train)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size,
                                num_workers=num_workers, shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=batch_size,
                                 num_workers=num_workers, shuffle=False) \
                                 if get_test else None
        return train_loader, val_loader, test_loader

    def load_datasets(self):
        raise NotImplementedError


class LabeledDocuments(Data):

    def __init__(self, file_path):
        super().__init__(file_path=file_path)

    def load_datasets(self):
        dataset = scipy.io.loadmat(self.file_path)

        # (num documents) x (vocab size) tensors containing tf-idf values
        Y_train = torch.from_numpy(dataset['train'].toarray()).float()
        Y_val = torch.from_numpy(dataset['cv'].toarray()).float()
        Y_test = torch.from_numpy(dataset['test'].toarray()).float()

        # (num documents) x (num labels) tensors containing {0,1}
        L_train = torch.from_numpy(dataset['gnd_train']).float()
        L_val = torch.from_numpy(dataset['gnd_cv']).float()
        L_test = torch.from_numpy(dataset['gnd_test']).float()

        self.train_dataset = TensorDataset(Y_train, L_train)
        self.val_dataset = TensorDataset(Y_val, L_val)
        self.test_dataset = TensorDataset(Y_test, L_test)

        self.vocab_size = self.train_dataset[0][0].size(0)
        self.num_labels = self.train_dataset[0][1].size(0)


class ArticlePairs(Data):

    def __init__(self, file_path):
        super().__init__(file_path=file_path)

    def load_datasets(self):
        # Each pair are dicts of form {word index: tf-idf value}
        (train_pairs, val_pairs, test_pairs, self.vocab) \
            = pickle.load(open(self.file_path, 'rb'))

        self.train_dataset = ArticlePairDataset(train_pairs, len(self.vocab))
        self.val_dataset = ArticlePairDataset(val_pairs, len(self.vocab))
        self.test_dataset = ArticlePairDataset(test_pairs, len(self.vocab))
        self.vocab_size = len(self.vocab)


class ArticlePairDataset(Dataset):
    def __init__(self, dict_pairs, vocab_size):
        self.dict_pairs = dict_pairs
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.dict_pairs)

    def __getitem__(self, index):
        article_X, article_Y = self.dict_pairs[index]
        return self.vectorize(article_X), self.vectorize(article_Y)

    def vectorize(self, article):
        u = torch.zeros(self.vocab_size)
        for i, weight in article.items():
            u[i] = weight
        return u
