"""Component timings including validation, allocation and synchronization."""
import argparse
import json
import statistics
import time

import torch
import vitni_torch

parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--repetitions", type=int, default=12)
args = parser.parse_args()
torch.set_num_threads(8)
records = []
for device in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
    def sync():
        if device == "mps": torch.mps.synchronize()
    for m,k,n in [(4,256,16),(32,512,64),(1,2048,2048)]:
        gen = torch.Generator().manual_seed(817)
        a = torch.randn(m,k,generator=gen).to(device)
        b = torch.randn(k,n,generator=gen).to(device)
        observations = {"vitni": [], "torch": []}
        first = {}
        for mode,fn in [("vitni",vitni_torch.canonical_matmul),("torch",torch.matmul)]:
            sync();start=time.perf_counter();fn(a,b);sync();first[mode]=time.perf_counter()-start
        for rep in range(args.repetitions):
            for mode in (["vitni","torch"] if rep%2==0 else ["torch","vitni"]):
                fn = vitni_torch.canonical_matmul if mode=="vitni" else torch.matmul
                sync();start=time.perf_counter();fn(a,b);sync()
                observations[mode].append(time.perf_counter()-start)
        records.append({"device":device,"shape":[m,k,n],"first_call_seconds":first,
                        "seconds":observations,
                        "median_seconds":{key:statistics.median(v)for key,v in observations.items()},
                        "vitni_over_torch_ratio":statistics.median(observations['vitni'])/statistics.median(observations['torch'])})
result={"torch":torch.__version__,"backend":vitni_torch.backend_info(),"repetitions":args.repetitions,
        "scope":"Unoptimized canonical float32 operator, includes validation/allocation/synchronization; not quantized full-model inference or the 2.47x figure. Native torch arithmetic can differ.",
        "records":records}
with open(args.output,"w")as out:json.dump(result,out,indent=2)
print(json.dumps(result,indent=2))
