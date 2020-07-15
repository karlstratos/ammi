import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Model
from pytorch_helper import get_init_function, FF
from vae import Decoder


class DVQ(Model):

    def __init__(self, hparams):
        super().__init__(hparams=hparams)

    def define_parameters(self):
        self.enc = FF(self.data.vocab_size, self.hparams.dim_hidden,
                      self.dim_dvq(), self.hparams.num_layers)

        self.dvq = DVQLayer(self.hparams.size_codebook,  # 2 for binary values
                            self.dim_codebook(),
                            self.hparams.num_features,  # num_splits
                            ema=self.hparams.ema, gamma=self.hparams.gamma,
                            alpha=self.hparams.alpha, beta=self.hparams.beta)
        self.dec = Decoder(self.dim_dvq(), self.data.vocab_size)

        self.apply(get_init_function(self.hparams.init))

    def dim_codebook(self):
        if self.hparams.dim_codebook < 0:
            dim_codebook = self.hparams.budget // (self.hparams.num_features *
                                                   self.data.vocab_size)
            if dim_codebook == 0:
                raise ValueError('Cannot allocate postive codebook dimension, '
                                 'increase memory budget')
        else:
            dim_codebook = self.hparams.dim_codebook

        return dim_codebook

    def dim_dvq(self):
        dim_dvq = self.hparams.num_features * self.dim_codebook()
        return dim_dvq

    def forward(self, Y, X=None):
        Y_encoded = self.enc(Y)  # B x num_features * dim_codebook
        dvq_output = self.dvq(Y_encoded)

        log_likelihood = self.dec(dvq_output['Z_embs'],
                                  (Y if X is None else X).sign())
        loss = -log_likelihood + dvq_output['loss']

        return {'loss': loss, 'log_likelihood': log_likelihood,
                'loss_dvq': dvq_output['loss']}

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.hparams.lr)]

    def configure_gradient_clippers(self):
        return [(self.parameters(), self.hparams.clip)]

    def encode_discrete(self, Y):
        return self.dvq(self.enc(Y))['argmins']

    def get_hparams_grid(self):
        grid = Model.get_general_hparams_grid()
        grid.update({
            'lr': [0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001],
            'batch_size': [16, 32, 64, 128],
            'dim_hidden': [100, 200],
            'num_layers': [0, 1],
            'ema': [False, True],
            'gamma': [0.99, 0.999],
            'alpha': [1e-5, 1e-7, 1e-9],
            'beta': [0.1, 0.25, 0.5, 1, 2, 4],
        })
        return grid

    @staticmethod
    def get_model_specific_argparser():
        parser = Model.get_general_argparser()

        parser.add_argument('--budget', type=int, default=40000000,
                            help='memory budget: if dim_codebook is unspecified'
                            ' (-1) allocate it so that # decoder params <= '
                            'budget [%(default)d]')
        parser.add_argument('--dim_codebook', type=int, default=50,
                            help='dimension of codebook embeddings (-1 auto) '
                            '[%(default)d]')
        parser.add_argument('--size_codebook', type=int, default=2,
                            help='number of codebook embeddings [%(default)d]')
        parser.add_argument('--ema', action='store_true',
                            help='use EMA?')
        parser.add_argument('--gamma', type=float, default=0.99,
                            help='retention rate for moving average '
                            '[%(default)g]')
        parser.add_argument('--alpha', type=float, default=1e-7,
                            help='Laplace smoothing [%(default)g]')
        parser.add_argument('--beta', type=float, default=0.1,
                            help='commitment loss weight [%(default)g]')

        return parser


# VQ-VAE: https://arxiv.org/pdf/1711.00937.pdf
class VQLayer(nn.Module):

    def __init__(self, size_codebook, dim_codebook, ema=True, gamma=0.99,
                 alpha=1e-9, beta=0.25):
        super().__init__()
        self.size_codebook = size_codebook  # (aka. K)
        self.dim_codebook = dim_codebook
        self.ema = ema
        self.gamma = gamma  # Retention rate for moving average
        self.alpha = alpha  # Laplace smoothing
        self.beta = beta  # Weight for commitment loss

        self.E = nn.Embedding(self.size_codebook, self.dim_codebook)

        if self.ema:
            self.register_buffer('cluster_sizes',
                                 torch.zeros(self.size_codebook))
            self.register_buffer('moving_avg', torch.Tensor(self.size_codebook,
                                                            self.dim_codebook))
            self.moving_avg.data = self.E.weight.clone()

    def forward(self, X):
        X_sqnorm = (X ** 2).sum(dim=1, keepdim=True)            # B x 1
        E_sqnorm = (self.E.weight ** 2).sum(dim=1).view(1, -1)  # 1 x K
        dist = X_sqnorm + E_sqnorm - 2 * X @ self.E.weight.t()  # B x K

        min_dist, argmins = torch.min(dist, dim=1)
        Z_embs = self.E(argmins)

        loss = self.beta * ((X - Z_embs.detach()) ** 2).sum(1).mean()
        if self.ema:
            self.update_E_ema(X, argmins)
        else:
            loss += ((X.detach() - Z_embs) ** 2).sum(1).mean()

        return {'Z_embs': X + (Z_embs - X).detach(), 'loss': loss,
                'argmins': argmins, 'min_dist': min_dist}

    def update_E_ema(self, X, argmins):
        if not self.training:
            return

        with torch.no_grad():
            one_hots = F.one_hot(argmins, self.size_codebook).float()  # B x K

            self.cluster_sizes = self.gamma * self.cluster_sizes + \
                                 (1 - self.gamma) * one_hots.sum(0)

            # Laplace smoothing
            self.cluster_sizes = (self.cluster_sizes + self.alpha) / \
                                 (1 + self.alpha * self.size_codebook /
                                  self.cluster_sizes.sum())

            self.moving_avg = self.gamma * self.moving_avg + \
                              (1 - self.gamma) * (one_hots.t() @ X)

            self.E.weight.data.copy_(self.moving_avg /
                                     self.cluster_sizes.unsqueeze(1))


class DVQLayer(nn.Module):
    def __init__(self, size_codebook, dim_codebook, num_splits, ema=True,
                 gamma=0.99, alpha=1e-9, beta=0.25):
        super().__init__()
        self.dim_codebook = dim_codebook
        self.num_splits = num_splits
        self.vqs = nn.ModuleList([VQLayer(size_codebook, dim_codebook,
                                          ema=ema, gamma=gamma, alpha=alpha,
                                          beta=beta)
                                  for _ in range(num_splits)])

    def forward(self, X):
        assert X.size(1) == self.num_splits * self.dim_codebook
        X_splits  = X.split(self.dim_codebook, dim=1)

        # Quantize each (B x dim_codebook) split.
        vq_out_list = [vq(X_split) for (X_split, vq) in zip(X_splits, self.vqs)]

        aggregate_out = {k: [vq_out[k] for vq_out in vq_out_list]
                         for k in vq_out_list[0].keys()}

        # Concat argmin codebook embeddings (connected to X by straight-through)
        # to get output with same dimension (B x num_splits * dim_codebook).
        Z_embs = torch.cat(aggregate_out['Z_embs'], dim=1)

        loss = torch.stack(aggregate_out['loss']).sum()  # 1

        argmins = torch.stack(aggregate_out['argmins'], dim=1)  # B x num_splits

        return {'Z_embs': Z_embs, 'loss': loss, 'argmins': argmins}


if __name__ == '__main__':
    argparser = DVQ.get_model_specific_argparser()
    hparams = argparser.parse_args()
    model = DVQ(hparams)
    if hparams.train:
        model.run_training_sessions()
    else:
        model.load()
        print('Loaded model with: %s' % model.flag_hparams())

        val_perf, test_perf = model.run_test()
        print('Val:  {:8.2f}'.format(val_perf))
        print('Test: {:8.2f}'.format(test_perf))
