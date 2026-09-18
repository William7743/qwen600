"""Whole-vocabulary metrics for the fixed V0-relative logits contract."""
import numpy as np


def logit_metrics(base,candidate):
    a=base.astype(np.float64);b=candidate.astype(np.float64)
    pa=np.exp(a-a.max());pa/=pa.sum()
    pb=np.exp(b-b.max());pb/=pb.sum()
    return {'mean_absolute_error':float(np.abs(a-b).mean()),
            'max_absolute_error':float(np.abs(a-b).max()),
            'probability_total_variation':float(np.abs(pa-pb).sum()/2),
            'baseline_top1_regret':float(a.max()-a[int(b.argmax())]),
            'baseline_top1':int(a.argmax()),'candidate_top1':int(b.argmax())}
