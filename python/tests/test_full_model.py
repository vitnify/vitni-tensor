import os
import errno
import shutil
from pathlib import Path
import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from vitni_torch import VitniModel

GGUF=os.environ.get('VITNI_FULL_MODEL_GGUF')
if os.environ.get('VITNI_REQUIRE_FULL_MODEL')=='1' and not GGUF:
    raise RuntimeError('VITNI_FULL_MODEL_GGUF is required; skipped model tests do not establish conformance')
pytestmark=pytest.mark.skipif(not GGUF,reason='requires real GGUF via VITNI_FULL_MODEL_GGUF')
CPU_ONLY=os.environ.get('VITNI_FULL_MODEL_CPU_ONLY')=='1'
if os.environ.get('VITNI_REQUIRE_METAL')=='1' and (CPU_ONLY or sys.platform!='darwin'):
    raise RuntimeError('Native Metal execution is required; CPU-only tests cannot satisfy this gate')
BACKENDS=['cpu']+(['metal'] if sys.platform=='darwin' and not CPU_ONLY else [])

def equal(a,b):assert torch.equal(a.view(torch.int32),b.view(torch.int32))

@pytest.fixture(scope='module',params=BACKENDS)
def model(request):
    with VitniModel(GGUF,backend=request.param,optimized=request.param=='metal',model_id='full-model-test') as value:
        yield value

@pytest.fixture(scope='module')
def cpu():
    with VitniModel(GGUF,backend='cpu',model_id='full-model-test') as value:yield value

def test_all_logits_and_receipts(model,cpu):
    tokens=torch.tensor([1,2,3])
    equal(model(tokens),cpu(tokens))
    assert model.generate(tokens,n_new=4)==cpu.generate(tokens,n_new=4)
    assert model.info()['model_loads']==1

def test_resets_errors_and_output_ownership(model,cpu):
    tokens=torch.tensor([1,2,3])
    expected=cpu(tokens)
    for other in [torch.tensor([7]),torch.tensor([1,9]),tokens]:
        equal(model(other),cpu(other))
    returned=model(tokens);returned.zero_()
    equal(model(tokens),expected)
    for invalid in [torch.tensor([-1]),torch.tensor([model.info()['vocab_size']]),torch.tensor([2**32])]:
        with pytest.raises(RuntimeError,match='vocabulary'):model(invalid)
    for invalid in [torch.tensor([]),torch.tensor([[1,2]]),torch.tensor([1.,2.]),torch.arange(8)[::2],torch.ones(513,dtype=torch.int64)]:
        with pytest.raises((ValueError,TypeError)):model(invalid)
    with pytest.raises(ValueError,match='context'):model.generate(tokens,n_new=512)
    with pytest.raises(ValueError):model.generate(tokens,n_new=-1)
    equal(model(tokens),expected)

def test_parallel_call_serialization(model,cpu):
    tokens=torch.tensor([1,2,3])
    expected=cpu(tokens)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for value in pool.map(lambda _:model(tokens),range(4)):equal(value,expected)

def test_explicit_model_contract(model):
    assert not model.training
    assert model.eval() is model
    with pytest.raises(RuntimeError,match='inference'):model.train()
    with pytest.raises(RuntimeError,match='backend'):model.to('cpu')
    with pytest.raises(RuntimeError,match='state_dict'):model.state_dict()
    with pytest.raises(RuntimeError,match='serialized'):pickle.dumps(model)

def test_no_subprocess_during_inference(model,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('inference must call the resident native library')
    monkeypatch.setattr(subprocess,'run',forbidden)
    assert model(torch.tensor([1,2])).shape==(2,model.info()['vocab_size'])
    assert len(model.generate(torch.tensor([1]),n_new=1)['tokens'])==1

def test_full_model_rejects_ambient_flush_mode(model):
    tokens=torch.tensor([1,2])
    expected=model(tokens)
    try:
        if not torch.set_flush_denormal(True):pytest.skip('host does not support flush control')
        with pytest.raises(RuntimeError,match='environment'):model(tokens)
        with pytest.raises(RuntimeError,match='environment'):model.generate(tokens,n_new=1)
    finally:torch.set_flush_denormal(False)
    equal(model(tokens),expected)

def test_close_and_context_manager():
    model=VitniModel(GGUF)
    model.close();model.close()
    with pytest.raises(RuntimeError,match='closed'):model(torch.tensor([1]))
    with pytest.raises(RuntimeError,match='closed'):model.info()

def test_no_ambient_optimization_mutation(monkeypatch):
    keys=['VITNI_METAL_Q4K_VARIANT','VITNI_METAL_RMS_VARIANT','VITNI_METAL_Q6K_COOPERATIVE',
          'VITNI_METAL_ATTENTION_COOPERATIVE','VITNI_METAL_CONCURRENT_DISPATCH']
    for key in keys:monkeypatch.setenv(key,'deliberately-invalid')
    before={key:os.environ[key]for key in keys}
    backend=BACKENDS[-1]
    with VitniModel(GGUF,backend=backend,optimized=backend=='metal') as model:
        assert model(torch.tensor([1])).shape[0]==1
        info=model.info()
        if backend=='metal':
            assert info['q4_variant']=='metadata' and info['rms_variant']=='no_contract'
            assert info['concurrent_dispatch'] and info['attention_cooperative'] and info['q6_cooperative']
    assert before=={key:os.environ[key]for key in keys}

def test_missing_model_and_backend_rejected(tmp_path):
    with pytest.raises(RuntimeError,match='read GGUF'):VitniModel(tmp_path/'missing.gguf')
    bad=tmp_path/'invalid.gguf';bad.write_bytes(b'bad')
    with pytest.raises(RuntimeError,match='parse GGUF'):VitniModel(bad)
    with pytest.raises(ValueError):VitniModel(GGUF,backend='cuda')
    with pytest.raises(ValueError):VitniModel(GGUF,backend='cpu',optimized=True)

@pytest.mark.parametrize('backend',BACKENDS)
def test_weight_file_not_reopened(backend,tmp_path):
    local=tmp_path/'resident.gguf'
    try:os.link(GGUF,local)
    except OSError as error:
        if error.errno!=errno.EXDEV:raise
        shutil.copyfile(GGUF,local)
    with VitniModel(local,backend=backend,optimized=backend=='metal') as model:
        local.unlink()
        assert model(torch.tensor([1,2])).shape[0]==2
        assert model.generate(torch.tensor([1]),n_new=1)['tokens']
