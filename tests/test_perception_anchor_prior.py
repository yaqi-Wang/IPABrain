import numpy as np
import torch
from src.data.retrieval_augmentation import aggregate_retrieval_priors
from src.utils.perception_anchor_prior import aggregate_anchor_parameters


def test_legacy_weighting_and_repeat_grouping():
    means=torch.tensor([[[0.]],[[2.]],[[8.]]])
    logvars=torch.tensor([[[2.]],[[4.]],[[10.]]])
    mu,lv,info=aggregate_anchor_parameters(means,logvars,[10,10,20],[0.75,0.75,0.25])
    torch.testing.assert_close(mu,torch.tensor([[2.75]]))
    torch.testing.assert_close(lv,torch.tensor([[4.75]]))
    assert info['image_ids']==[10,20] and info['trials_per_image']==[2,1]
    old_mu,old_lv,_=aggregate_retrieval_priors(np.array([[0,1]]),np.array([[.75,.25]]),
                                               torch.tensor([[[1.]],[[8.]]]),torch.tensor([[[3.]],[[10.]]]))
    torch.testing.assert_close(mu,old_mu[0]);torch.testing.assert_close(lv,old_lv[0])


def test_rejects_invalid_inputs():
    x=torch.zeros(2,1,1)
    for ids,sims in [([1,1],[.5,.6]),([1,2],[-.5,0]),([1,2],[.5,float('nan')])]:
        try:aggregate_anchor_parameters(x,x,ids,sims)
        except ValueError:pass
        else:raise AssertionError('Invalid anchor input was accepted')


if __name__=='__main__':
    test_legacy_weighting_and_repeat_grouping();test_rejects_invalid_inputs()
    print('Legacy anchor weighting and validation checks passed')
