import pytest
import torch
from psg_only.models import B1Model
from psg_only.windows import context_options
from psg_only.train import _checkpoint
from psg_only.evaluate import load_model


def test_80_context_slice_and_reload(tmp_path):
    model=B1Model(hidden_dim=4,layers=1,dropout=0.,input_epochs=80).eval()
    x=torch.randn(2,80,192)
    with torch.no_grad():
        full,_=model.lstm(model.norm(x))
        expected=model.head(full[:,30:50])
        assert torch.equal(model(x),expected)
    cfg={'model':{'hidden_dim':4,'layers':1,'dropout':0.},'data':{'input_epochs':80,'target_epochs':20,'left_context':30,'stride':20}}
    path=tmp_path/'checkpoint.pt';_checkpoint(path,model,cfg,'b1',0,0.)
    restored,_=load_model(path,torch.device('cpu'))
    with torch.no_grad():
        assert torch.equal(restored(x),expected)
    with pytest.raises(ValueError,match='shape'):
        restored(torch.zeros(2,40,192))


@pytest.mark.parametrize('data',[{'input_epochs':81},{'input_epochs':80,'left_context':10},{'input_epochs':10},{'input_epochs':80,'stride':40}])
def test_invalid_context_rejected(data):
    with pytest.raises(ValueError):
        context_options(data)
