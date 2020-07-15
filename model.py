import argparse
import math
import random
import torch
import torch.nn as nn

from collections import OrderedDict
from copy import deepcopy
from data import ArticlePairs, LabeledDocuments
from datetime import timedelta
from evaluation import compute_retrieval_precision, compute_matching_accuracy
from logger import Logger
from timeit import default_timer as timer


class Model(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.load_data()

    def load_data(self):
        if 'document_hashing' in self.hparams.data_path:
            self.data = LabeledDocuments(self.hparams.data_path)
            self.multiview = False
        elif 'related_articles' in self.hparams.data_path:
            self.data = ArticlePairs(self.hparams.data_path)
            self.multiview = True
        else:
            raise ValueError('data_path must indicate task type')

    def define_parameters(self):
        raise NotImplementedError

    def configure_optimizers(self):
        raise NotImplementedError

    def configure_gradient_clippers(self):
        raise NotImplementedError

    def encode_discrete(self, target_inputs):
        raise NotImplementedError

    def run_training_sessions(self):
        logger = Logger(self.hparams.model_path + '.log', on=True)
        val_perfs = []
        best_val_perf = float('-inf')
        start = timer()
        random.seed(self.hparams.seed)  # For reproducible random runs

        for run_num in range(1, self.hparams.num_runs + 1):
            state_dict, val_perf = self.run_training_session(run_num, logger)
            val_perfs.append(val_perf)

            if val_perf > best_val_perf:
                best_val_perf = val_perf
                logger.log('----New best {:8.2f}, saving'.format(val_perf))
                torch.save({'hparams': self.hparams,
                            'state_dict': state_dict}, self.hparams.model_path)

        logger.log('Time: %s' % str(timedelta(seconds=round(timer() - start))))
        self.load()
        if self.hparams.num_runs > 1:
            logger.log_perfs(val_perfs)
            logger.log('best hparams: ' + self.flag_hparams())

        val_perf, test_perf = self.run_test()
        logger.log('Val:  {:8.2f}'.format(val_perf))
        logger.log('Test: {:8.2f}'.format(test_perf))

    def run_training_session(self, run_num, logger):
        self.train()

        # Scramble hyperparameters if number of runs is greater than 1.
        if self.hparams.num_runs > 1:
            logger.log('RANDOM RUN: %d/%d' % (run_num, self.hparams.num_runs))
            for hparam, values in self.get_hparams_grid().items():
                assert hasattr(self.hparams, hparam)
                self.hparams.__dict__[hparam] = random.choice(values)

        random.seed(self.hparams.seed)
        torch.manual_seed(self.hparams.seed)

        self.define_parameters()
        logger.log(str(self))
        logger.log('%d params' % sum([p.numel() for p in self.parameters()]))
        logger.log('hparams: %s' % self.flag_hparams())

        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        self.to(device)

        optimizers = self.configure_optimizers()
        gradient_clippers = self.configure_gradient_clippers()
        train_loader, val_loader, _ = self.data.get_loaders(
            self.hparams.batch_size, self.hparams.num_workers,
            shuffle_train=True, get_test=False)
        best_val_perf = float('-inf')
        best_state_dict = None
        forward_sum = {}
        num_steps = 0
        bad_epochs = 0

        try:
            for epoch in range(1, self.hparams.epochs + 1):
                for batch_num, batch in enumerate(train_loader):
                    for optimizer in optimizers:
                        optimizer.zero_grad()

                    if self.multiview:
                        X = batch[0].to(device)
                        Y = batch[1].to(device)
                    else:
                        X = None
                        Y = batch[0].to(device)

                    if self.hparams.no_tfidf:
                        X = X.sign() if self.multiview else None
                        Y = Y.sign()

                    forward = self.forward(Y, X=X)
                    for key in forward:
                        if key in forward_sum:
                            forward_sum[key] += forward[key]
                        else:
                            forward_sum[key] = forward[key]
                    num_steps += 1

                    if (batch_num + 1) % self.hparams.check_interval == 0:
                        logger.log('Epoch {:3d} | batch {:5d}/{:5d}'.format(
                            epoch, batch_num + 1, len(train_loader)), False)
                        logger.log(' '.join([' | {:s} {:8.2f}'.format(
                            key, forward_sum[key] / num_steps)
                                             for key in forward_sum]))

                    if math.isnan(forward_sum['loss']):
                        logger.log('Stopping epoch because loss is NaN')
                        break

                    forward['loss'].backward()

                    for params, clip in gradient_clippers:
                        nn.utils.clip_grad_norm_(params, clip)

                    for optimizer in optimizers:
                        optimizer.step()

                if math.isnan(forward_sum['loss']):
                    logger.log('Stopping training session because loss is NaN')
                    break

                val_perf = self.evaluate(train_loader, val_loader, device)
                logger.log('End of epoch {:3d}'.format(epoch), False)
                logger.log(' '.join([' | {:s} {:8.2f}'.format(
                    key, forward_sum[key] / num_steps)
                                     for key in forward_sum]), False)
                logger.log(' | val perf {:8.2f}'.format(val_perf), False)

                if val_perf > best_val_perf:
                    best_val_perf = val_perf
                    bad_epochs = 0
                    logger.log('\t\t*Best model so far, deep copying*')
                    best_state_dict = deepcopy(self.state_dict())
                else:
                    bad_epochs += 1
                    logger.log('\t\tBad epoch %d' % bad_epochs)

                if bad_epochs > self.hparams.num_bad_epochs:
                    break

        except KeyboardInterrupt:
            logger.log('-' * 89)
            logger.log('Exiting from training early')
            pass

        return best_state_dict, best_val_perf

    def load(self):
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        checkpoint = torch.load(self.hparams.model_path) if self.hparams.cuda \
                     else torch.load(self.hparams.model_path,
                                     map_location=torch.device('cpu'))
        if checkpoint['hparams'].cuda and not self.hparams.cuda:
            checkpoint['hparams'].cuda = False
        self.hparams = checkpoint['hparams']
        self.define_parameters()
        self.load_state_dict(checkpoint['state_dict'])
        self.to(device)

    def evaluate(self, train_loader, eval_loader, device):
        self.eval()
        with torch.no_grad():
            if self.multiview:  # Matching acc in article pairs in eval
                perf = compute_matching_accuracy(eval_loader, device,
                                                 self.encode_discrete,
                                                 self.hparams.distance_metric,
                                                 self.hparams.num_retrieve)
            else:  # Retrieval prec of eval wrt labeled docs in train
                perf = compute_retrieval_precision(train_loader, eval_loader,
                                                   device, self.encode_discrete,
                                                   self.hparams.distance_metric,
                                                   self.hparams.num_retrieve)
        self.train()
        return perf

    def run_test(self):
        device = torch.device('cuda' if self.hparams.cuda else 'cpu')
        train_loader, val_loader, test_loader \
            = self.data.get_loaders(128, self.hparams.num_workers,
                                    shuffle_train=False, get_test=True)

        val_perf = self.evaluate(train_loader, val_loader, device)
        test_perf = self.evaluate(train_loader, test_loader, device)
        return val_perf, test_perf

    def flag_hparams(self):
        flags = '%s %s' % (self.hparams.model_path, self.hparams.data_path)
        for hparam in vars(self.hparams):
            val = getattr(self.hparams, hparam)
            if str(val) == 'False':
                continue
            elif str(val) == 'True':
                flags += ' --%s' % (hparam)
            elif str(hparam) in {'model_path', 'data_path', 'num_runs',
                                 'num_workers'}:
                continue
            else:
                flags += ' --%s %s' % (hparam, val)
        return flags

    @staticmethod
    def get_general_hparams_grid():
        grid = OrderedDict({
            'seed': list(range(100000)),
            'lr': [0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001],
            'clip': [1, 5, 10],
            'batch_size': [16, 32, 64, 128, 256],
            'init': [0, 0.5, 0.1, 0.05, 0.01],
            })
        return grid

    @staticmethod
    def get_general_argparser():
        parser = argparse.ArgumentParser()

        parser.add_argument('model_path', type=str)
        parser.add_argument('data_path', type=str)
        parser.add_argument('--train', action='store_true',
                            help='train a model?')
        parser.add_argument('--num_features', type=int, default=16,
                            help='num discrete features [%(default)d]')
        parser.add_argument('--dim_hidden', type=int, default=24,
                            help='dimension of hidden state [%(default)d]')
        parser.add_argument('--num_layers', type=int, default=0,
                            help='num layers [%(default)d]')
        parser.add_argument('--batch_size', type=int, default=16,
                            help='batch size [%(default)d]')
        parser.add_argument('--lr', type=float, default=0.01,
                            help='initial learning rate [%(default)g]')
        parser.add_argument('--init', type=float, default=0.1,
                            help='unif init range (default if 0) [%(default)g]')
        parser.add_argument('--clip', type=float, default=10,
                            help='gradient clipping [%(default)g]')
        parser.add_argument('--epochs', type=int, default=40,
                            help='max number of epochs [%(default)d]')
        parser.add_argument('--num_runs', type=int, default=1,
                            help='num random runs (not random if 1) '
                            '[%(default)d]')
        parser.add_argument('--check_interval', type=int, default=1000,
                            help='number of updates for check [%(default)d]')
        parser.add_argument('--no_tfidf', action='store_true',
                            help='raw bag-of-words as input instead of tf-idf?')
        parser.add_argument('--distance_metric', default='hamming',
                            choices=['hamming', 'cosine'],
                            help='distance metric [%(default)s]')
        parser.add_argument('--num_retrieve', type=int, default=100,
                            help='num neighbors to retrieve [%(default)d]')
        parser.add_argument('--num_bad_epochs', type=int, default=6,
                            help='num indulged bad epochs [%(default)d]')
        parser.add_argument('--num_workers', type=int, default=0,
                            help='num dataloader workers [%(default)d]')
        parser.add_argument('--seed', type=int, default=9061,
                            help='random seed [%(default)d]')
        parser.add_argument('--cuda', action='store_true',
                            help='use CUDA?')

        return parser
