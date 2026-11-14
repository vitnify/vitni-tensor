"""Workload-specific predictors for Andreas Lindeman's unchanged llmzip.py.

No Vitni source or upstream coder/probability conversion is modified.
"""
from pathlib import Path
import importlib.util
import json
import os
import subprocess
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM
from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

OUT=Path(__file__).resolve().parent
SOURCE=OUT/'cache/upstream'
spec=importlib.util.spec_from_file_location('independent_llmzip',SOURCE/'llmzip.py')
up=importlib.util.module_from_spec(spec);spec.loader.exec_module(up)
MODELS=json.loads((OUT/'cache/models.json').read_text())
HF=MODELS['hf_snapshot'];GGUF=MODELS['gguf']
HF_ID='HuggingFaceTB/SmolLM2-135M'
Q8_HASH=MODELS['files']['SmolLM2-135M-Q8_0.gguf']['sha256']
PROFILE_IDS={'hf':HF_ID,'gguf':f'gguf-smol135-q8:{Q8_HASH}',
             'vitni':f'vitni-v033-r2-smol135-q8:{Q8_HASH}'}


class NativePredictor(up.Predictor):
    def __init__(self,provider,device,window=256,keep=64):
        started=time.perf_counter()
        if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('physical MPS required')
        torch.set_num_threads(10)
        if provider=='hf':
            tok_loader=AutoTokenizer.from_pretrained
            model_loader=AutoModelForCausalLM.from_pretrained
            with patch.object(AutoTokenizer,'from_pretrained',side_effect=lambda *a,**k:tok_loader(HF,local_files_only=True)), \
                 patch.object(AutoModelForCausalLM,'from_pretrained',side_effect=lambda *a,**k:model_loader(HF,dtype=torch.float32,local_files_only=True)):
                super().__init__(HF_ID,device=device,dtype='float32',window=window,keep=keep,threads=10,quiet=True)
        else:
            parsed=load_gguf_checkpoint(GGUF)
            cfg=LlamaConfig(**parsed['config']);cfg._attn_implementation='sdpa'
            with torch.device('meta'):model=LlamaForCausalLM(cfg)
            converted=load_gguf_checkpoint(GGUF,return_tensors=True,model_to_load=model,torch_dtype=torch.float32)
            state=converted['tensors']
            # GGUF may omit a tied classifier. Preserve the exact embedding alias;
            # do not invent a separate matrix or load unrelated HF weights.
            self.tied_classifier_alias=False
            if cfg.tie_word_embeddings and 'lm_head.weight' not in state:
                state['lm_head.weight']=state['model.embed_tokens.weight']
                self.tied_classifier_alias=True
            model.load_state_dict(state,assign=True,strict=True)
            model.model.rotary_emb=LlamaRotaryEmbedding(cfg)
            model.to(device=device,dtype=torch.float32).eval()
            self.torch=torch;self.tok=AutoTokenizer.from_pretrained(HF,local_files_only=True)
            self.model=model;self.model_id=PROFILE_IDS['gguf'];self.device=device;self.dtype='float32'
            self.window=window;self.keep=keep;self.seed=self._seed_ids()
            self.tokens=[];self.past=None;self.cache_len=0;self._logits=None
        self.load_seconds=time.perf_counter()-started
        self.forward_seconds=0.;self.forward_positions=0;self.resets=0
        self.provider=provider

    def _forward(self,ids,past):
        start=time.perf_counter();result=super()._forward(ids,past)
        if self.device=='mps':torch.mps.synchronize()
        self.forward_seconds+=time.perf_counter()-start;self.forward_positions+=len(ids)
        if past is None:self.resets+=1
        return result

    def close(self):
        del self.model


class VitniPredictor(up.Predictor):
    def __init__(self,device,window=256,keep=64,log=None):
        if device not in ('cpu','metal'):raise ValueError('explicit cpu or metal required')
        started=time.perf_counter()
        self.torch=torch;self.tok=AutoTokenizer.from_pretrained(HF,local_files_only=True)
        self.model_id=PROFILE_IDS['vitni'];self.device='vitni-canonical-r2';self.dtype='float32'
        self.window=window;self.keep=keep;self.seed=self._seed_ids()
        self.tokens=[];self.past=None;self.cache_len=0;self._logits=None
        self.forward_seconds=0.;self.forward_positions=0;self.resets=0
        self.actual_backend=device
        self.error_log=Path(log or OUT/f'vitni-{device}-stderr.log').open('wb')
        self.process=subprocess.Popen([str(OUT/'native/target/release/vitni-workload-predictor'),GGUF,device,str(window)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.error_log)
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Vitni worker did not start; inspect stderr')
        self.ready=json.loads(line)
        assert self.ready['backend']==device and self.ready['vocab']==49152
        self.load_seconds=time.perf_counter()-started

    @property
    def vocab_size(self):return self.ready['vocab']

    def _forward(self,ids,past):
        operation='P' if past is None else 'S'
        command=operation+' '+','.join(map(str,ids))+'\n'
        self.process.stdin.write(command.encode());self.process.stdin.flush()
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Vitni worker failed; no fallback')
        header=json.loads(line);raw=self.process.stdout.read(header['words']*4)
        if len(raw)!=header['words']*4:raise RuntimeError('truncated native output')
        row=np.frombuffer(raw,dtype='<f4').copy()
        self.forward_seconds+=header['seconds'];self.forward_positions+=header['positions']
        if past is None:self.resets+=1
        return SimpleNamespace(logits=torch.from_numpy(row).reshape(1,1,-1),past_key_values=True)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write(b'QUIT\n');self.process.stdin.flush();self.process.wait(timeout=30)
        self.error_log.close()


def predictor(provider,device,window=256,keep=64,log=None):
    if provider=='vitni':return VitniPredictor(device,window,keep,log)
    return NativePredictor(provider,device,window,keep)
