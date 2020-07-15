import argparse
import torch
import torch.nn as nn

from model import Model
from pytorch_helper import get_init_function, FF


# Mostly BMSH: https://www.aclweb.org/anthology/D19-1526.pdf
class VAE(Model):

    def __init__(self, hparams):
        super().__init__(hparams=hparams)

    def define_parameters(self):
        self.enc = BerEncoder(self.data.vocab_size, self.hparams.dim_hidden,
                              self.hparams.num_features,
                              self.hparams.num_layers)
        self.dec = Decoder(self.hparams.num_features, self.data.vocab_size)
        self.logvar_mlp = FF(self.hparams.num_features,
                             self.hparams.dim_hidden,
                             self.hparams.num_features, 1)
        self.cenc = CatEncoder(self.data.vocab_size, self.hparams.dim_hidden,
                               self.hparams.num_components,
                               self.hparams.num_layers)

        if self.multiview:
            self.gamma = nn.Linear(self.data.vocab_size,
                                   self.hparams.num_components *
                                   self.hparams.num_features)
            self.pc = nn.Linear(self.data.vocab_size,
                                self.hparams.num_components)
        else:
            self.gamma = nn.Embedding(self.hparams.num_components,
                                      self.hparams.num_features)
            self.pc = nn.Embedding(1, self.hparams.num_components)

        self.apply(get_init_function(self.hparams.init))

    def forward(self, Y, X=None):
        q1_Y = self.enc(Y)
        Z = torch.bernoulli(q1_Y)
        Z_st = q1_Y + (Z - q1_Y).detach()

        stdev = 0.5 * self.logvar_mlp(q1_Y).exp()
        Z_st = Z_st + torch.randn_like(Z_st) * stdev  # data-dependent noise

        log_likelihood = self.dec(Z_st, (Y if X is None else X).sign())
        kl = self.compute_kl(Y, q1_Y)

        loss = -log_likelihood + self.hparams.beta * kl

        return {'loss': loss, 'log_likelihood': log_likelihood, 'kl': kl}

    def compute_kl(self, Y, q1_Y):
        if self.multiview:
            pC = self.pc(Y).softmax(dim=1)
            p1_C = self.gamma(Y).view(Y.size(0),
                                      self.hparams.num_components,
                                      self.hparams.num_features)
            p1_C = p1_C.clamp(min=-1, max=1)  # Can be unstable
            p1_C = p1_C.sigmoid()
        else:
            pC = self.pc.weight.softmax(dim=1).repeat(Y.size(0), 1)
            p1_C = self.gamma.weight.sigmoid().expand(
                Y.size(0), self.hparams.num_components,
                self.hparams.num_features)

        qC_Y = self.cenc(Y)
        klC = (qC_Y * (qC_Y.log() - pC.log())).sum(1).mean()

        q1_Y = q1_Y.unsqueeze(1).expand(Y.size(0),
                                        self.hparams.num_components, -1)
        q0_Y = 1 - q1_Y
        klZ_C = (q1_Y * (q1_Y.log() - p1_C.log()) +
                 q0_Y * (q0_Y.log() - (1 - p1_C).log())).sum(2)
        klZ = (qC_Y * klZ_C).sum(1).mean()

        return klC + klZ

    def configure_optimizers(self):
        return [torch.optim.Adam(self.parameters(), lr=self.hparams.lr)]

    def configure_gradient_clippers(self):
        return [(self.parameters(), self.hparams.clip)]

    def encode_discrete(self, Y):
        return self.enc(Y).round()

    def get_hparams_grid(self):
        grid = Model.get_general_hparams_grid()
        grid.update({
            'lr': [0.003, 0.001, 0.0003, 0.0001, 0.00003, 0.00001],
            'dim_hidden': [300, 400, 500, 600, 700],
            'num_components': [10, 20, 40, 80],
            'num_layers': [0, 1, 2],
            'beta': [1, 2, 3],
            })
        return grid

    @staticmethod
    def get_model_specific_argparser():
        parser = Model.get_general_argparser()

        parser.add_argument('--num_components', type=int, default=20,
                            help='num mixture components [%(default)d]')
        parser.add_argument('--beta', type=float, default=1,
                            help='beta term (as in beta-VAE) [%(default)g]')

        return parser


class BerEncoder(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, num_layers):
        super().__init__()
        self.ff = FF(dim_input, dim_hidden, dim_output, num_layers)

    def forward(self, Y):
        return torch.sigmoid(self.ff(Y))


class CatEncoder(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, num_layers):
        super().__init__()
        self.ff = FF(dim_input, dim_hidden, dim_output, num_layers)

    def forward(self, Y):
        return self.ff(Y).softmax(dim=1)


class Decoder(nn.Module):  # As in VDSH, NASH, BMSH
    def __init__(self, dim_encoding, vocab_size):
        super().__init__()
        self.E = nn.Embedding(dim_encoding, vocab_size)
        self.b = nn.Parameter(torch.zeros(1, vocab_size))

    def forward(self, Z, targets):  # (B x m), (B x V binary)
        scores = Z @ self.E.weight + self.b # B x V
        log_probs = scores.log_softmax(dim=1)
        log_likelihood = (log_probs * targets).sum(1).mean()
        return log_likelihood


if __name__ == '__main__':
    argparser = VAE.get_model_specific_argparser()
    hparams = argparser.parse_args()
    model = VAE(hparams)
    if hparams.train:
        model.run_training_sessions()
    else:
        model.load()
        print('Loaded model with: %s' % model.flag_hparams())

        val_perf, test_perf = model.run_test()
        print('Val:  {:8.2f}'.format(val_perf))
        print('Test: {:8.2f}'.format(test_perf))
