import torch
import torch.nn as nn

def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]

def sample_negatives(y, n_negatives=5):

    num = y.size(1)
    #n_negatives = 5
    cross_sample_negatives = 0

    if n_negatives == 0 and cross_sample_negatives == 0:
        return y.new(0)

    bsz, tsz, fsz = y.shape
    y = y.view(-1, fsz)  # BTC => (BxT)C

    cross_high = tsz * bsz
    high = tsz
    with torch.no_grad():
        assert high > 1, f"{bsz,tsz,fsz}"

        if n_negatives > 0:
            tszs = (
                buffered_arange(num)
                .unsqueeze(-1)
                .expand(-1, n_negatives)
                .flatten()
            )

            neg_idxs = torch.randint(
                low=0, high=high - 1, size=(bsz, n_negatives * num)
            )
            neg_idxs[neg_idxs >= tszs] += 1

        if cross_sample_negatives > 0:
            tszs = (
                buffered_arange(num)
                .unsqueeze(-1)
                .expand(-1, cross_sample_negatives)
                .flatten()
            )

            cross_neg_idxs = torch.randint(
                low=0,
                high=cross_high - 1,
                size=(bsz, cross_sample_negatives * num),
            )
            cross_neg_idxs[cross_neg_idxs >= tszs] += 1

    if n_negatives > 0:
        for i in range(1, bsz):
            neg_idxs[i] += i * high
    else:
        neg_idxs = cross_neg_idxs

    if cross_sample_negatives > 0 and n_negatives > 0:
        neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

    negs = y[neg_idxs.view(-1)]
    negs = negs.view(
        bsz, num, n_negatives + cross_sample_negatives, fsz
    ).permute(
        2, 0, 1, 3
    )  # to NxBxTxC
    return negs, neg_idxs
