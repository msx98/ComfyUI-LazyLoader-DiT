#!/usr/bin/env python3
"""Manual MPS test plan.  Do not run this from the node installer."""
import argparse, gc, os, sys, time
import psutil, torch

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--comfy", required=True, help="ComfyUI checkout to import")
parser.add_argument("--height", type=int, default=128)
parser.add_argument("--width", type=int, default=128)
parser.add_argument("--limits-mib", type=float, nargs="+", default=[32768.0, 16384.0, 8192.0])
args = parser.parse_args(); sys.path.insert(0, args.comfy)
from streaming import load_lazy_diffusion_model
import comfy.sd

def mps_counters():
    mps = getattr(torch, "mps", None); out = {}
    for n in ("current_allocated_memory", "driver_allocated_memory", "recommended_max_memory"):
        f = getattr(mps, n, None)
        if callable(f):
            try: out[n] = f()
            except Exception: pass
    return out
def make_inputs(patcher):
    dm = patcher.model.diffusion_model; dtype = patcher.model.get_dtype_inference(); dev = patcher.load_device
    x = torch.randn(1, getattr(dm, "channels", 16), args.height, args.width, device=dev, dtype=dtype)
    if dm.__class__.__name__ == "SingleStreamDiT":
        context = torch.randn(1, 8, dm.txtlayers * dm.txtdim, device=dev, dtype=dtype); t = torch.tensor([1.], device=dev)
        return (x, t, context, {})
    context = torch.randn(1, 8, dm.params.context_in_dim, device=dev, dtype=dtype); t = torch.tensor([1.], device=dev)
    return (x, t, context, {"pooled_output": torch.zeros(1, dm.params.vec_in_dim, device=dev, dtype=dtype), "guidance": torch.tensor([3.5], device=dev)})
def one(patcher, inputs):
    x, t, context, extra = inputs
    return patcher.model.apply_model(x, t, c_crossattn=context, **extra)
stock = comfy.sd.load_diffusion_model(args.checkpoint)
import comfy.model_management
comfy.model_management.load_model_gpu(stock)
torch.mps.synchronize()
inputs = make_inputs(stock)
reference = one(stock, inputs).detach().float().cpu()
del stock; gc.collect()

results = {}
for limit_mib in args.limits_mib:
    gc.collect(); before = psutil.Process().memory_info().rss
    p = load_lazy_diffusion_model(args.checkpoint, {}, limit_mib, True); torch.mps.synchronize()
    loaded = psutil.Process().memory_info().rss; y = one(p, inputs); torch.mps.synchronize()
    start = time.perf_counter()
    for _ in range(3): y = one(p, inputs)
    torch.mps.synchronize(); step = (time.perf_counter() - start) / 3
    results[limit_mib] = (y.detach().float().cpu(), loaded - before, step, mps_counters())
    print(f"limit={limit_mib:.0f} MiB: RSS delta={loaded-before:,}; sec/step={step:.3f}; MPS={results[limit_mib][3]}")
for limit_mib, (output, *_rest) in results.items():
    torch.testing.assert_close(reference, output, rtol=2e-3, atol=2e-3)
print("each lazy budget matches the stock loader")
