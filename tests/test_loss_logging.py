import torch
from psg_only.train import WeightedLossMeter


def test_weighted_loss_aggregation_matches_single_batch():
    weights=torch.tensor([1.,2.,3.,4.])
    logits=torch.tensor([[2.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.],[0.,0.,0.,0.]])
    targets=torch.tensor([0,1,2,3,-100])
    meter=WeightedLossMeter(weights)
    for lo,hi in [(0,1),(1,4),(4,5)]:
        meter.add_logits(logits[lo:hi],targets[lo:hi])
    expected=torch.nn.functional.cross_entropy(logits,targets,weight=weights,ignore_index=-100)
    assert abs(meter.mean()-expected.item()) < 1e-6
    empty=WeightedLossMeter(weights);empty.add_logits(logits[-1:],targets[-1:])
    assert empty.mean() is None
