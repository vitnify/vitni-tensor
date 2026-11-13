"""Opt-in eager nn.Module over a resident instance of the actual full engine."""
from __future__ import annotations
import ctypes
from functools import lru_cache
import json
import os
from pathlib import Path
import threading
import weakref

import torch
from ._ops import _cpu_lib


@lru_cache(maxsize=1)
def _library():
    lib=_cpu_lib()
    try:
        lib.vitni_model_last_error.restype=ctypes.c_char_p
    except AttributeError as error:
        raise RuntimeError('Rebuild the research binding; this native library has no full-model API') from error
    pointer,size=ctypes.c_void_p,ctypes.c_size_t
    for name,args in {
        'open':[pointer,size,ctypes.c_uint32,ctypes.c_uint32,pointer],
        'close':[ctypes.c_uint64],
        'info':[ctypes.c_uint64,pointer,size,pointer],
        'forward':[ctypes.c_uint64,pointer,size,pointer,size,pointer],
        'generate':[ctypes.c_uint64,pointer,size,size,pointer,size,pointer,size,pointer,pointer],
    }.items():
        function=getattr(lib,'vitni_model_'+name)
        function.argtypes=args
        function.restype=ctypes.c_int
    return lib


def _check(lib,code):
    if code:
        raise RuntimeError(lib.vitni_model_last_error().decode('utf8',errors='replace'))


def _release(lib,handle,pid):
    if os.getpid()==pid:lib.vitni_model_close(handle)


class VitniModel(torch.nn.Module):
    """Unbatched, inference-only GGUF model; CPU token IDs and CPU float32 logits.

    backend='metal' executes the existing resident MetalForward, not PyTorch MPS
    shaders. Each forward/generate begins a fresh sequence and reuses allocations.
    This eager bridge does not promise torch.compile/export or HF compatibility.
    """
    def __init__(self,gguf:str|Path,*,backend:str='cpu',optimized:bool=False,
                 model_id:str='vitni-torch-full-model'):
        super().__init__()
        if backend not in ('cpu','metal'):raise ValueError('backend must be cpu or metal')
        if optimized and backend!='metal':raise ValueError('optimized is a Metal-only preset')
        if not isinstance(model_id,str):raise TypeError('model_id must be a string')
        self._lock=threading.RLock()
        self._lib=_library()
        self._pid=os.getpid()
        self._handle=0
        self.model_id=model_id
        path=str(Path(gguf).resolve()).encode('utf8')
        handle=ctypes.c_uint64()
        _check(self._lib,self._lib.vitni_model_open(path,len(path),int(backend=='metal'),int(optimized),ctypes.byref(handle)))
        self._handle=handle.value
        self._finalizer=weakref.finalize(self,_release,self._lib,self._handle,self._pid)
        self._config=self.info()
        self.last_native_seconds=None
        super().train(False)

    def _ensure(self):
        if os.getpid()!=self._pid:raise RuntimeError('Reopen VitniModel in the child process; handles cannot cross fork')
        if not self._handle:raise RuntimeError('VitniModel is closed')

    def _tokens(self,tokens):
        if not isinstance(tokens,torch.Tensor):raise TypeError('input_ids must be a torch.Tensor')
        if tokens.device.type!='cpu' or tokens.dtype!=torch.int64 or tokens.layout!=torch.strided:
            raise TypeError('input_ids must be a CPU int64 strided tensor')
        if tokens.ndim!=1 or not tokens.is_contiguous():raise ValueError('input_ids must be a contiguous rank-1 tensor')
        if not 1<=tokens.numel()<=self._config['max_context']:
            raise ValueError('input_ids exceed the nonempty context bound')

    @torch.compiler.disable
    def forward(self,input_ids:torch.Tensor)->torch.Tensor:
        with self._lock:
            self._ensure();self._tokens(input_ids)
            result=torch.empty((input_ids.numel(),self._config['vocab_size']),dtype=torch.float32,device='cpu')
            seconds=ctypes.c_double()
            _check(self._lib,self._lib.vitni_model_forward(self._handle,input_ids.data_ptr(),input_ids.numel(),
                result.data_ptr(),result.numel(),ctypes.byref(seconds)))
            self.last_native_seconds=seconds.value
            return result

    def generate(self,input_ids:torch.Tensor,*,n_new:int=16)->dict:
        """Run the existing engine's generation/certificate driver; return its receipt dict."""
        with self._lock:
            self._ensure();self._tokens(input_ids)
            if isinstance(n_new,bool) or not isinstance(n_new,int) or n_new<0:
                raise ValueError('n_new must be a nonnegative integer')
            if input_ids.numel()+n_new>self._config['max_context']:
                raise ValueError('prompt plus generation exceeds context bound')
            buffer=ctypes.create_string_buffer(16384)
            written=ctypes.c_size_t();seconds=ctypes.c_double()
            model_id=self.model_id.encode('utf8')
            _check(self._lib,self._lib.vitni_model_generate(self._handle,input_ids.data_ptr(),input_ids.numel(),n_new,
                model_id,len(model_id),buffer,len(buffer),ctypes.byref(written),ctypes.byref(seconds)))
            self.last_native_seconds=seconds.value
            return json.loads(buffer.raw[:written.value])

    def info(self)->dict:
        with self._lock:
            self._ensure()
            buffer=ctypes.create_string_buffer(16384);written=ctypes.c_size_t()
            _check(self._lib,self._lib.vitni_model_info(self._handle,buffer,len(buffer),ctypes.byref(written)))
            return json.loads(buffer.raw[:written.value])

    def close(self):
        with self._lock:
            if not self._handle:return
            self._ensure()
            self._finalizer();self._handle=0

    def __enter__(self):self._ensure();return self
    def __exit__(self,*args):self.close()
    def __getstate__(self):raise RuntimeError('Native engine handles cannot be serialized; reopen the GGUF explicitly')
    def state_dict(self,*args,**kwargs):raise RuntimeError('Native GGUF weights are not a PyTorch state_dict')
    def _apply(self,*args,**kwargs):raise RuntimeError('Select backend when opening VitniModel; tensor to()/dtype conversion is unsupported')
    def train(self,mode=True):
        if mode:raise RuntimeError('VitniModel is inference-only')
        return super().train(False)
