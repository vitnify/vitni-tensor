"""Importing the package must not activate the full model or replace Torch ops."""
import subprocess
import sys


def test_import_keeps_full_model_opt_in():
    subprocess.run([sys.executable,'-c',r'''
import os
import torch
matmul = torch.matmul
linear = torch.nn.Linear.forward
deterministic = torch.are_deterministic_algorithms_enabled()
environment = {key:value for key,value in os.environ.items() if key.startswith('VITNI_')}
import vitni_torch
from vitni_torch import _model
assert torch.matmul is matmul
assert torch.nn.Linear.forward is linear
assert torch.are_deterministic_algorithms_enabled() == deterministic
assert {key:value for key,value in os.environ.items() if key.startswith('VITNI_')} == environment
assert _model._library.cache_info().currsize == 0, 'import must not open the native full-model API'
'''],check=True)
