# Author: Karl Stratos (karlstratos@gmail.com)

"""
NOTATION

      B = batch size
      m = number of binary variables z=(z_1, ..., z_m) in {0,1}^m
      o = Markov order of p
      r = Markov order of q >= o

      P = B x m x 2^(o+1) tensor where P[b,i,:] tabulates 2^(o+1) cond probs
          p(z_i|y[b], i, z_{i-o}, ..., z_{i-1}) as (omitting y[b] and i)

            p(0|00...0) p(0|10...0) p(0|01...0) p(0|11...0) ... p(0|11...1)
              p(1|00...0) p(1|10...0) p(1|01...0) p(1|11...0) ... p(1|11...1)

      Q = B x m x 2^(r+1) tensor with q(z_i|x[b], i, z_{i-r}, ..., z_{i-1})
     P_ = P in logits
     Q_ = Q in logits
"""

import itertools
import math
import numpy as np
import torch
import torch.nn.functional as F


def estimate_hZ_X(P, Q_, quads=None, buffers=[]):
    """
           P: [0,1]^{B x m x 2^(o+1)}
          Q_:     R^{B x m x 2^(r+1)}
       quads: (k0, l0, k1, l1)^{2^o}
     buffers: list of one-hot tensors {0,1}^{B x 1 x 2^(o+i)} for i=1...r-o
    """
    o = int(math.log2(P.size(2)) - 1)
    r = int(math.log2(Q_.size(2)) - 1)
    assert r >= o

    if r > o and (not buffers or buffers[0].size(0) != P.size(0)):
        buffers = precompute_buffers(P.size(0), o, r, P.device)

    if o == 0:
        marginal = P
    else:
        Pi = compute_forward(P, quads)     # B x m x 2^o
        marginal = Pi.repeat(1, 1, 2) * P  # B x m x 2^(o+1)

    for i in range(r - o):
        marginal_shifted = torch.cat([buffers[i], marginal], dim=1)[:,:-1,:]
        marginal = marginal_shifted.repeat(1, 1, 2) * \
                   P.repeat_interleave(pow(2, i + 1),
                                       dim=2)  # B x m x 2^(o+1+i+1)

    hZ_X = (- (marginal * F.logsigmoid(Q_)).sum(dim=(1, 2))).mean()
    return hZ_X  # >= 0


def estimate_hZ_X_brute(P, Q):
    """
     P: [0,1]^{B x m x 2^(o+1)}
     Q: [0,1]^{B x m x 2^(r+1)}
    """
    contributions = []
    for z in itertools.product([0, 1], repeat=P.size(1)):
        p = compute_prob(P, z)
        q = compute_prob(Q, z)
        contributions.append(- p * q.log())
    hZ_X = torch.stack(contributions, dim=1).sum(dim=1).mean()
    return hZ_X  # R


def compute_viterbi(P, quads=None):
    """
         P: [0,1]^{B x m x 2^(o+1)}
     quads: (k0, l0, k1, l1)^{2^o}
    """
    B = P.size(0)
    m = P.size(1)
    o = int(math.log2(P.size(2)) - 1)

    if o == 0:
        zs = torch.round(P[:,:,1])  # B x m
        max_probs = P.gather(2, zs.unsqueeze(2).long())\
                     .prod(dim=1).squeeze(1)
        return zs, max_probs

    if not isinstance(quads, torch.Tensor):
        quads = precompute_quads(o).to(P.device)

    slices = [torch.zeros(B, pow(2, o)).to(P.device)]
    slices[0][:, 0].fill_(1)  # Always ends in 0...0 at position 0.
    bp = torch.zeros(B, m + 1, pow(2, o)).long().to(P.device)  # Backpointers

    for i in range(1, m + 1):
        slice = []
        for j, (k0, l0, k1, l1) in enumerate(quads):
            from0 = slices[-1][:, k0] * P[:, i - 1, l0]
            from1 = slices[-1][:, k1] * P[:, i - 1, l1]
            values01 = torch.stack([from0, from1], dim=1)  # B x 2
            stick, vec01 = torch.max(values01, 1)
            slice.append(stick)

            # Convert 0-1s back to indices.
            K = torch.LongTensor([k0, k1]).repeat(B, 1).to(P.device)  # B x 2
            bp[:, i, j] = K.gather(1, vec01.unsqueeze(1)).squeeze(1)
        slices.append(torch.stack(slice, dim=1))

    max_probs, indices = torch.max(slices[-1], 1)  # This is actually (m+1)-th

    tape = []
    tape.append([np.binary_repr(k, width=o)[::-1] for k in indices])
    for i in range(m, o, -1):  # m ... o + 1
        indices = bp[:, i, :].gather(1, indices.unsqueeze(1).long()).squeeze(1)
        tape.append([np.binary_repr(k, width=o)[-1] for k in indices])

    zs = []
    for b in range(B):
        bitstring = ''.join(tape[i][b] for i in reversed(range(len(tape))))
        zs.append([int(bit) for bit in bitstring[-m:]])  # We might have o > m
    zs = torch.tensor(zs).float().to(P.device)

    return zs, max_probs  # {0,1}^{B x m}, [0,1]^B


def compute_viterbi_brute(P):
    """
     P: [0,1]^{B x m x 2^(o+1)}
    """
    zs = [None for _ in range(P.size(0))]
    max_probs = [0.0 for _ in range(P.size(0))]
    for z in itertools.product([0, 1], repeat=P.size(1)):
        prob = compute_prob(P, z)
        for b in range(prob.size(0)):
            if prob[b].item() > max_probs[b]:
                max_probs[b] = prob[b].item()
                zs[b] = z
    zs = torch.tensor(zs).float().to(P.device)
    max_probs = torch.tensor(max_probs).to(P.device)
    return zs, max_probs  # {0,1}^{B x m}, [0,1]^B


def estimate_hZ(P, I=None):
    """
     P: [0,1]^{B x m x 2^(o+1)}
     I: index^{B x m x 2^m}

     This is still O(2^m) but prob faster than brute.
    """
    if not isinstance(I, torch.Tensor):
        I = precompute_I(P.size(1), int(math.log2(P.size(2)) - 1)).\
            repeat(P.size(0), 1, 1).to(P.device)
    pZ = P.gather(2, I).prod(dim=1).mean(dim=0)  # [0,1]^{2^m}
    hZ = (-pZ * torch.log(pZ)).sum()
    return hZ  # R


def estimate_hZ_brute(P):
    """
     P: [0,1]^{B x m x 2^(o+1)}
    """
    contributions = []
    for z in itertools.product([0, 1], repeat=P.size(1)):
        pZ = compute_prob(P, z).mean()
        contributions.append(- pZ * pZ.log())
    hZ = torch.stack(contributions).sum()
    return hZ  # R


def compute_prob(P, z):
    """
     P: [0,1]^{B x m x 2^(o+1)}
     z: {0,1}^m  (list or tuple, not a tensor)
    """
    o = int(math.log2(P.size(2)) - 1)
    z = [0 for _ in range(o)] + list(z)
    probs = []
    for i in range(o, len(z)):
        j = int(''.join(map(str, z[i - o:i + 1]))[::-1], 2)
        probs.append(P[:, i - o, j])
    pZ_Y = torch.stack(probs, dim=1).prod(dim=1)
    return pZ_Y  # [0,1]^B


def compute_forward(P, quads=None):
    """
         P: [0,1]^{B x m x 2^(o+1)}
     quads: (k0, l0, k1, l1)^{2^o}

    OUTPUT

      Pi = B x m x 2^o tensor where Pi[b][i] tabulates 2^o probs
          pi(z_{i-o}...z_{i-1}|y[b], i-1) defined as sum of probs of seqs of
          length i-1 ending with z_{i-o}...z_{i-1} as (omitting y[b] and i-1)

               pi(00..0) pi(10..0) pi(01..0) ... pi(11..1)
    """
    B = P.size(0)
    m = P.size(1)
    o = int(math.log2(P.size(2)) - 1)

    if not isinstance(quads, torch.Tensor):
        quads = precompute_quads(o).to(P.device)

    slices = [torch.zeros(B, pow(2, o)).to(P.device)]
    slices[0][:, 0].fill_(1)  # Always ends in 0...0 at position 0.
    for i in range(1, m):
        slice = []
        for j, (k0, l0, k1, l1) in enumerate(quads):
            slice.append(slices[-1][:, k0] * P[:, i - 1, l0] + \
                         slices[-1][:, k1] * P[:, i - 1, l1])
        slices.append(torch.stack(slice, dim=1))
    Pi = torch.stack(slices, dim=1)

    return Pi  # [0,1]^{B x m x 2^o}


def precompute_quads(o):
    """
     quads: list of index quadruples such that under our convention

           quads[j][0] = index of j-th     (z_{i-o}=0...z_i)
           quads[j][1] = index of j-th (z_i|z_{i-o}=0...z_i)
           quads[j][2] = index of j-th     (z_{i-o}=1...z_i)
           quads[j][3] = index of j-th (z_i|z_{i-o}=1...z_i)
    """
    zs = [''.join(map(str, bits)) for bits in itertools.product([0, 1],
                                                                repeat=o)]
    quads = [(int(z[1:] + '0', 2), int(z + '0', 2),
              int(z[1:] + '1', 2), int(z + '1', 2)) for z in zs]
    return torch.LongTensor(quads)


def precompute_I(m, o):
    """
     I: {0...2^(o+1)-1}^{m x 2^m} such that

             P[i, I[i,j]] = p(z_i|i, z_{i-o}...z_{i-1}) using j-th z in {0,1}^m
    """
    assert o <= m

    zs = torch.LongTensor(list(itertools.product([0, 1], repeat=m))).t().flip(0)
    if o > 0:
        zs = torch.cat((torch.zeros(o, pow(2, m)).long(), zs), dim=0)
    I = torch.zeros(m, pow(2, m))
    for i in range(o, o + m):
        for j in range(pow(2, m)):
            I[i - o, j] = int(''.join(
                map(str, zs[i - o:i + 1, j].flip(0).tolist())), 2)

    return I.long()


def precompute_buffers(B, o, r, device):
    buffers = []
    for i in range(r - o):
        buffer = torch.zeros(B, 1, pow(2, o + 1 + i)).to(device)
        buffer[:, 0, 0].fill_(1)
        buffers.append(buffer)
    return buffers
