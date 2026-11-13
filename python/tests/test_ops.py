import os
import ctypes
import platform
from pathlib import Path

import numpy as np
import pytest
import torch

import vitni_torch as vt
from vitni_torch import _ops

MPS = torch.backends.mps.is_available()
if os.environ.get("VITNI_REQUIRE_MPS") == "1" and not MPS:
    raise RuntimeError("A real MPS device is required; do not count skips as validation")
DEVICES = ["cpu"] + (["mps"] if MPS else [])


def bits_equal(a, b):
    assert a.shape == b.shape
    assert torch.equal(a.cpu().view(torch.int32), b.cpu().view(torch.int32))


def tree(values):
    values = list(values)
    while len(values) > 1:
        values = [np.float32(values[i] + values[i+1]) if i+1 < len(values) else values[i]
                  for i in range(0, len(values), 2)]
    return values[0] if values else np.float32(0)


def dot(a, b):
    chunks = []
    for base in range(0, len(a), 8192):
        lanes = [np.float32(0) for _ in range(8)]
        for i in range(base, min(base+8192, len(a))):
            product = np.float32(a[i] * b[i])
            j = (i-base) % 8
            lanes[j] = np.float32(lanes[j] + product)
        chunks.append(tree(lanes))
    return tree(chunks)


def raw_q4(k, m, seed=7):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 256, (m, k//256*144), dtype=np.uint8)
    for i, block in enumerate(data.reshape(-1, 144)):
        d = [0x3001, 0x3403, 0x0001, 0x8001, 0x0400, 0x8000][i % 6]
        block[0:2] = np.frombuffer(int(d).to_bytes(2, 'little'), dtype=np.uint8)
        block[2:4] = np.frombuffer(int(0x2803).to_bytes(2, 'little'), dtype=np.uint8)
    return torch.from_numpy(data)


def dequant(row):
    output = []
    for block in row.reshape(-1, 144):
        d, dmin = [np.float32(np.frombuffer(bytes(block[i:i+2]), dtype='<f2')[0]) for i in [0,2]]
        scales, qs = block[4:16], block[16:]
        def sm(j):
            if j < 4:
                return int(scales[j]) & 63, int(scales[j+4]) & 63
            return ((int(scales[j+4]) & 15) | ((int(scales[j-4]) >> 6) << 4),
                    (int(scales[j+4]) >> 4) | ((int(scales[j]) >> 6) << 4))
        for sub in range(4):
            for segment in range(2):
                scale, minimum = sm(2*sub+segment)
                ds = np.float32(d * np.float32(scale))
                dm = np.float32(dmin * np.float32(minimum))
                for t in range(32):
                    q = int(qs[32*sub+t])
                    q = q & 15 if segment == 0 else q >> 4
                    output.append(np.float32(np.float32(ds*np.float32(q))-dm))
    return np.array(output, dtype=np.float32)


@pytest.mark.parametrize("shape", [(0,0,0),(2,0,3),(0,7,3),(2,7,0),(1,1,1),
    (3,7,2),(2,8,3),(3,9,4),(2,8191,2),(2,8192,2),(2,8193,2),(1,16640,2),(1,27648,2)])
def test_matmul_exact_cpu_mps_and_independent_order(shape):
    m,k,n = shape
    gen = torch.Generator().manual_seed(k+17)
    a, b = torch.randn(m,k,generator=gen), torch.randn(k,n,generator=gen)
    if a.numel() >= 8:
        edges = torch.tensor([0, -(2**31), 0x2f800000, 0xaf800000-2**32, 0x4f800000, 0x3f800001, 0x3f7ffffe, 0xbf800000-2**32],dtype=torch.int32).view(torch.float32)
        a.flatten()[:8] = edges
    reference = np.empty((m,n), dtype=np.float32)
    av,bv = a.numpy(),b.numpy()
    for i in range(m):
        for j in range(n): reference[i,j] = dot(av[i],bv[:,j])
    reference = torch.from_numpy(reference)
    cpu = vt.canonical_matmul(a,b)
    bits_equal(cpu,reference)
    if MPS:
        aa,bb = a.to('mps'),b.to('mps')
        for _ in range(2):bits_equal(vt.canonical_matmul(aa,bb),reference)


@pytest.mark.parametrize("k,m", [(256,0),(256,1),(768,3),(2048,9),(8192,2),(8448,5),(16640,1)])
def test_q4_exact_cpu_mps_and_independent_decode(k,m):
    x = torch.randn(k,generator=torch.Generator().manual_seed(11))
    x[0],x[1] = 0.0,-0.0
    weights = raw_q4(k,m)
    reference = torch.tensor([dot(x.numpy(),dequant(row)) for row in weights.numpy()],dtype=torch.float32)
    bits_equal(vt.q4_k_linear(x,weights),reference)
    if MPS:
        xx,ww = x.to('mps'),weights.to('mps')
        for _ in range(2):bits_equal(vt.q4_k_linear(xx,ww),reference)


@pytest.mark.parametrize("device", DEVICES)
def test_nonzero_storage_offsets(device):
    a = torch.randn(25).to(device)[1:].reshape(3,8)
    b = torch.randn(17).to(device)[1:].reshape(8,2)
    result = vt.canonical_matmul(a,b)
    bits_equal(result,vt.canonical_matmul(a.cpu(),b.cpu()))
    x = torch.randn(257).to(device)[1:]
    weights = torch.cat([torch.zeros(3,dtype=torch.uint8),raw_q4(256,3).flatten()]).to(device)[3:].reshape(3,144)
    bits_equal(vt.q4_k_linear(x,weights),vt.q4_k_linear(x.cpu(),weights.cpu()))


@pytest.mark.parametrize("device", DEVICES)
def test_opcheck_both_operators(device):
    examples = [(torch.ops.vitni.canonical_matmul.default,(torch.randn(2,9,device=device),torch.randn(9,3,device=device))),
                (torch.ops.vitni.q4_k_linear.default,(torch.randn(768,device=device),raw_q4(768,3).to(device)))]
    for op,args in examples:
        result = torch.library.opcheck(op,args)
        assert all(v == "SUCCESS" for v in result.values())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend", ["eager","aot_eager","inductor"])
def test_compile_exact(device,backend):
    a,b = torch.randn(2,9,device=device),torch.randn(9,3,device=device)
    compiled = torch.compile(vt.canonical_matmul,backend=backend,fullgraph=True)
    bits_equal(compiled(a,b),vt.canonical_matmul(a,b))
    x,w = torch.randn(256,device=device),raw_q4(256,3).to(device)
    compiled = torch.compile(vt.q4_k_linear,backend=backend,fullgraph=True)
    bits_equal(compiled(x,w),vt.q4_k_linear(x,w))


@pytest.mark.parametrize("device", DEVICES)
def test_export_reload(tmp_path,device):
    class Matmul(torch.nn.Module):
        def forward(self,a,b): return vt.canonical_matmul(a,b)
    args=(torch.randn(2,9,device=device),torch.randn(9,3,device=device))
    exported=torch.export.export(Matmul(),args,strict=True)
    assert any(n.target==torch.ops.vitni.canonical_matmul.default for n in exported.graph.nodes)
    torch.export.save(exported,tmp_path/'matmul.pt2')
    restored=torch.export.load(tmp_path/'matmul.pt2').module()
    bits_equal(restored(*args),vt.canonical_matmul(*args))


@pytest.mark.parametrize("device", DEVICES)
def test_q4_export_reload(tmp_path,device):
    class Linear(torch.nn.Module):
        def forward(self,x,w): return vt.q4_k_linear(x,w)
    args=(torch.randn(256,device=device),raw_q4(256,3).to(device))
    exported=torch.export.export(Linear(),args,strict=True)
    assert any(n.target==torch.ops.vitni.q4_k_linear.default for n in exported.graph.nodes)
    torch.export.save(exported,tmp_path/'q4.pt2')
    restored=torch.export.load(tmp_path/'q4.pt2').module()
    bits_equal(restored(*args),vt.q4_k_linear(*args))


@pytest.mark.parametrize("device", DEVICES)
def test_reject_invalid_contract(device):
    a,b=torch.ones(2,8,device=device),torch.ones(8,3,device=device)
    with pytest.raises((TypeError,ValueError,RuntimeError)):vt.canonical_matmul(a.double() if device=='cpu' else a.half(),b)
    with pytest.raises((ValueError,RuntimeError)):vt.canonical_matmul(a.t(),torch.ones(2,3,device=device))
    with pytest.raises((ValueError,RuntimeError)):vt.canonical_matmul(a,torch.ones(9,3,device=device))
    with pytest.raises((ValueError,RuntimeError)):vt.q4_k_linear(torch.ones(255,device=device),raw_q4(256,1).to(device))
    with pytest.raises((ValueError,RuntimeError)):vt.q4_k_linear(torch.ones(256,device=device),torch.zeros(1,143,dtype=torch.uint8,device=device))
    with pytest.raises(RuntimeError,match='inference-only'):vt.canonical_matmul(a.requires_grad_(),b)
    a=a.detach()
    for value in [float('nan'),float('inf')]:
        bad=a.clone();bad[0,0]=value
        with pytest.raises((ValueError,RuntimeError),match='finite|invalid'):vt.canonical_matmul(bad,b)
    with pytest.raises((ValueError,RuntimeError),match='finite|overflow|magnitude'):
        vt.canonical_matmul(torch.full((1,1),torch.finfo(torch.float32).max,device=device),torch.full((1,1),2.,device=device))
    for word in [1,0x00800000,0x2f7fffff,0x4f800001]:
        value=torch.tensor([word],dtype=torch.int32).view(torch.float32).to(device).reshape(1,1)
        with pytest.raises(ValueError,match='magnitude'):
            vt.canonical_matmul(value,torch.ones(1,1,device=device))
    invalid_weights=raw_q4(256,1)
    invalid_weights[0,0],invalid_weights[0,1]=0,0x7c  # half +infinity
    with pytest.raises((ValueError,RuntimeError),match='finite|overflow'):
        vt.q4_k_linear(torch.ones(256,device=device),invalid_weights.to(device))
    with pytest.raises((ValueError,RuntimeError),match='64 canonical chunks'):
        vt.canonical_matmul(torch.empty(1,_ops.MAX_K+1,device=device),torch.empty(_ops.MAX_K+1,1,device=device))


@pytest.mark.skipif(not MPS,reason='requires actual MPS hardware')
def test_mps_has_no_rust_cpu_fallback(monkeypatch):
    a,b=torch.randn(2,7),torch.randn(7,3)
    x,w=torch.randn(256),raw_q4(256,3)
    expected_a,expected_x=vt.canonical_matmul(a,b),vt.q4_k_linear(x,w)
    monkeypatch.setattr(_ops,'_cpu_lib',lambda: (_ for _ in ()).throw(AssertionError('CPU fallback forbidden')))
    bits_equal(vt.canonical_matmul(a.to('mps'),b.to('mps')),expected_a)
    bits_equal(vt.q4_k_linear(x.to('mps'),w.to('mps')),expected_x)


def test_ambient_flush_to_zero_rejected():
    try:
        if not torch.set_flush_denormal(True):pytest.skip('host does not support flush control')
        with pytest.raises(RuntimeError,match='environment'):
            vt.canonical_matmul(torch.ones(1,1),torch.ones(1,1))
    finally:torch.set_flush_denormal(False)


def test_ambient_upward_rounding_rejected():
    libc=ctypes.CDLL(None)
    if not hasattr(libc,'fesetround'):pytest.skip('host does not expose fenv control')
    original=libc.fegetround()
    upward=0x00400000 if platform.machine().lower() in ('arm64','aarch64') else 0x0800
    # Prepare all numerical inputs before altering the ambient rounding mode.
    a,b=torch.ones(1,1),torch.ones(1,1)
    x,w=torch.ones(256),raw_q4(256,1)
    try:
        if libc.fesetround(upward)!=0:pytest.skip('host does not support tested upward mode')
        for function,arguments in [(vt.canonical_matmul,(a,b)),(vt.q4_k_linear,(x,w))]:
            with pytest.raises(RuntimeError,match='environment'):function(*arguments)
    finally:
        assert libc.fesetround(original)==0


def test_packaged_shaders_match_engine_sources():
    root=Path(__file__).resolve().parents[2]
    for name in ['canonical_matmul.metal','q4k.metal']:
        assert (Path(_ops.__file__).parent/'_shaders'/name).read_bytes()==(root/'kernels'/name).read_bytes()
