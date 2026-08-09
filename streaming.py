"""Safetensors-only resident-set pager for ComfyUI DiTs.

ARCHITECTURE TABLE -- keep checkpoint/layout facts here.  All names are from
the ComfyUI checkout audited on 2026-08-06, not guessed from upstream repos.

| Comfy class | target | ordered block lists | checkpoint prefixes | conversion |
| Flux | FLUX family / Flux2 dev / Flux2 klein | double_blocks, single_blocks | same | *_norm.scale -> *_norm.weight |
| SingleStreamDiT | FLUX.1 Krea [dev] | blocks | same | identity |
| QwenImageTransformer2DModel | Qwen-Image, Qwen-Image-Edit (2509/2511) | transformer_blocks | same | identity |

The conversion column is load-bearing, not documentation: see _DIT_ARCHITECTURES.
Qwen-Image is detected by `txt_norm.weight` and reuses one class across its
variants -- 2511 differs only by a `__index_timestep_zero__` marker key that
sets default_ref_method, and Layered by an `addition_t_embedding`.  None of
that changes the block list or the key space, so one row covers them all.

Flux.2 needs no separate row: model_detection sets image_model="flux2" from
`double_stream_modulation_img.lin.weight` but still builds
comfy.ldm.flux.model.Flux, and supported_models.Flux2 subclasses Flux, so it
inherits the same two block lists and the same process_unet_state_dict
rewrite.  The fp8 release differs only in storage -- see the quantization
table below.

Flux.forward_orig executes double_blocks then single_blocks.  Krea2
SingleStreamDiT._forward executes blocks.  `txtfusion.*` is deliberately not
paged: it is a non-main-transformer text-side projection and remains resident.

READS -- streamed block tensors are pread into one reusable buffer, never
mapped.  safetensors returns tensors that point into its mmap, so reading one
faults those pages in and the live mapping holds them resident; every forward
pass touches every block, so a 26 GB checkpoint left ~23 GB of resident mapped
pages that nothing released.  _DirectReader preads instead, with F_NOCACHE so
the kernel does not retain them either.  Resident non-block tensors are still
read through safetensors: they are read once, at load.

SLOT ALIASING -- a consequence of the design, not a bug to fix.  With k slots
and c = ceil(k/2) per section, slot (section, local) is shared by every block
whose index is congruent: blocks b and b+2c alternate through the same bytes,
so one storage address holds different weights at different points in a single
forward pass.  Any consumer that caches derived weight state keyed on
data_ptr() or on the storage -- rather than on the tensor object -- will serve
a conversion of the wrong block's numbers.  bind_chunk_block hands out a fresh
tensor object per bind, which defeats identity- and weakref-keyed caches; a
storage-keyed one cannot be defeated without one buffer per block, which is
exactly the memory this package exists to not spend.  ComfyUI's own
cast_bias_weight recasts per call and is unaffected.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import dataclasses
import json
import logging
import math
import fcntl
import os
import struct
import threading
import time
import traceback
import types
from collections.abc import MutableMapping

import torch
from safetensors import safe_open

import comfy.float
import comfy.lora
import comfy.model_detection
import comfy.model_management
import comfy.model_patcher
import comfy.sd
import comfy.utils


# Worker threads no longer submit anything to Metal, so no cross-thread fence
# is needed.  The earlier design had the worker issue copy_ into MPS slots and
# then call torch.mps.synchronize() from that thread; PyTorch's MPS backend
# encodes onto one command buffer, so that raced with the main thread's
# encoding and aborted inside -[IOGPUMetalCommandBuffer setCurrentCommandEncoder:]
# ("failed assertion _status < MTLCommandBufferStatusCommitted").  All device
# writes now happen on the thread running forward, in submission order, which
# is also what orders a slot's refill after the compute that last read it.
_PREADV = hasattr(os, "preadv")


class _ShapeOnlyStateDict(MutableMapping):
    """Mapping sufficient for ComfyUI's detector; values expose only .shape/.dtype."""
    def __init__(self, reader):
        self._reader = reader
        self._keys = list(reader.keys())
        self._shapes = {}
    def __getitem__(self, key):
        if key not in self._shapes:
            sl = self._reader.get_slice(key)
            self._shapes[key] = types.SimpleNamespace(shape=tuple(sl.get_shape()), dtype=None)
        return self._shapes[key]
    def __setitem__(self, key, value): raise TypeError("read only")
    def __delitem__(self, key): raise TypeError("read only")
    def __iter__(self): return iter(self._keys)
    def __len__(self): return len(self._keys)


class _TensorReader:
    """Logical UNet-key view of one safe_open handle (also supports full ckpts)."""
    def __init__(self, path):
        self.path = path
        self.raw = safe_open(path, framework="pt", device="cpu")
        physical = list(self.raw.keys())
        candidates = ("model.diffusion_model.", "model.model.", "net.")
        counts = {p: sum(k.startswith(p) for k in physical) for p in candidates}
        prefix = max(counts, key=counts.get) if max(counts.values(), default=0) > 5 else ""
        self.prefix = prefix
        self.keymap = {k[len(prefix):]: k for k in physical if k.startswith(prefix)} if prefix else {k: k for k in physical}
        self.ranges = self._read_ranges(path)
    @staticmethod
    def _read_ranges(path):
        """Absolute (offset, length) per physical key, from the safetensors header.

        safe_open exposes no offsets, and the header is a documented, stable
        part of the format: 8-byte little-endian length, then JSON whose
        data_offsets are relative to the end of that header.
        """
        try:
            with open(path, "rb") as handle:
                (length,) = struct.unpack("<Q", handle.read(8))
                header = json.loads(handle.read(length))
        except Exception:
            return {}
        start = 8 + length
        out = {}
        for key, info in header.items():
            if key == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            out[key] = (start + begin, end - begin, info["dtype"], tuple(info["shape"]))
        return out
    def byte_range(self, key):
        entry = self.ranges.get(self.keymap[key])
        return entry[:2] if entry else None
    def layout(self, key):
        """(offset, length, torch dtype, shape) or None if not derivable."""
        entry = self.ranges.get(self.keymap[key])
        if entry is None:
            return None
        offset, length, name, shape = entry
        dtype = _SAFETENSORS_DTYPES.get(name)
        return None if dtype is None else (offset, length, dtype, shape)
    def keys(self): return self.keymap.keys()
    def get_tensor(self, key): return self.raw.get_tensor(self.keymap[key])
    def get_slice(self, key): return self.raw.get_slice(self.keymap[key])
    def metadata(self): return self.raw.metadata()


def _set_qualified(module, name, value, *, buffer=False):
    parent_name, _, leaf = name.rpartition(".")
    parent = module.get_submodule(parent_name) if parent_name else module
    if buffer:
        parent._buffers[leaf] = value
    else:
        parent._parameters[leaf] = value


def _named_local_tensors(module):
    for name, tensor in module.named_parameters(recurse=True):
        yield name, True, tensor
    for name, tensor in module.named_buffers(recurse=True):
        yield name, False, tensor


def _checkpoint_key(model_key):
    # comfy.supported_models.Flux.process_unet_state_dict() is the only target
    # conversion in this checkout.  Keep it beside the table above.
    if model_key.endswith("_norm.weight"):
        return model_key[:-len(".weight")] + ".scale"
    return model_key


_MTL_STORAGE_MODE_SHARED = 0
_OBJC = None


def _objc_send(pointer, selector, restype):
    """objc_msgSend with an exact prototype -- arm64's variadic ABI needs one."""
    global _OBJC
    if _OBJC is None:
        library = ctypes.CDLL(ctypes.util.find_library("objc"))
        library.sel_registerName.restype = ctypes.c_void_p
        library.sel_registerName.argtypes = [ctypes.c_char_p]
        _OBJC = library
    send = ctypes.CDLL(None).objc_msgSend
    send.restype = restype
    send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return send(ctypes.c_void_p(pointer), ctypes.c_void_p(_OBJC.sel_registerName(selector)))


def _host_pointer(tensor):
    """CPU-writable address of an MPS tensor's storage, or None.

    On Apple Silicon PyTorch's MPS allocator hands out MTLResourceStorageMode
    Shared buffers, so "device" memory is addressable from the CPU.  A memmove
    into that address never enters the Metal queue -- which is the whole point
    here, because a queued copy_ serializes behind the compute already
    submitted and that wait is most of a chunk's cost.

    Every condition below is a reason the address would be wrong rather than
    slow, so all of them return None instead of guessing:
      - not MPS, or torch built without it
      - a non-zero storage_offset: data_ptr is then not the buffer's base
      - a buffer SMALLER than the tensor: the address cannot cover the write
      - storageMode other than Shared: there is no CPU-visible mapping

    A buffer *larger* than the tensor is normal and fine.  PyTorch's MPS
    allocator buckets allocation sizes, so almost every slot sits in a rounded-
    up buffer; requiring an exact match rejects nearly all of them.  What that
    check was really guarding against -- several tensors suballocated from one
    buffer, where writing at contents() would clobber a neighbour -- is caught
    properly by the overlap test in _check_async_capable, which compares the
    addresses themselves rather than inferring from a size.
    """
    if tensor.device.type != "mps" or tensor.storage_offset() != 0:
        return None
    wanted = tensor.numel() * tensor.element_size()
    try:
        handle = tensor.data_ptr()
        if not handle:
            return None
        if _objc_send(handle, b"storageMode", ctypes.c_ulong) != _MTL_STORAGE_MODE_SHARED:
            return None
        if _objc_send(handle, b"length", ctypes.c_ulong) < wanted:
            return None
        contents = _objc_send(handle, b"contents", ctypes.c_void_p)
    except Exception:
        return None
    return contents or None


def _host_write(address, source):
    """memmove a contiguous CPU tensor into a host-visible device address.

    Measured faster than preading straight into the same address (17.6 vs
    7.1 GiB/s): the kernel's copy-to-user into a Metal mapping is slower than
    a userspace copy into it, so reading into an ordinary buffer and moving
    from there beats fusing the two.
    """
    if not source.is_contiguous():
        source = source.contiguous()
    ctypes.memmove(address, source.data_ptr(), source.numel() * source.element_size())


def _synchronize_device(device):
    """Wait for the device queue to go idle. No-op off MPS/CUDA."""
    kind = getattr(device, "type", None)
    if kind == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif kind == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def _identity_key(model_key):
    """No conversion.  Correct for targets whose loader has no key rewrite.

    Callers that page a non-Flux target must pass this explicitly rather than
    inheriting the Flux rule: the rule rewrites *any* ``*_norm.weight``, which
    would silently mistranslate e.g. Qwen3's ``self_attn.q_norm.weight``.
    """
    return model_key


def _cpu_dequantize(source, dtype, scale=None):
    """Widen/rescale a CPU tensor to `dtype`, or hand it back if it already is.

    Split out of _dtype_copy_ so the off-queue path can reuse exactly the same
    arithmetic: memmove moves bytes and nothing else, so any conversion has to
    have happened before it.
    """
    if scale is None and source.dtype == dtype:
        return source
    with torch.inference_mode(False), torch.no_grad():
        # to(dtype) first, then scale, rather than via fp32: fp8_e4m3 carries
        # three mantissa bits and int8 needs seven, both inside bf16's eight,
        # so the widening itself is exact and this avoids a temporary four
        # times the tensor's size.
        out = source.to(dtype)
        # to() returns `source` itself when the dtype already matches, and a
        # tensor read through safetensors inside a loader node running under
        # inference_mode() is an inference tensor -- mutating it in place
        # raises.  Scale out-of-place in that case; in-place only when to()
        # already handed us a fresh tensor.
        if scale is not None and scale != 1.0:
            out = out.mul(scale) if out is source else out.mul_(scale)
    return out


def _dtype_copy_(destination, source, scale=None):
    # Source is one safetensors tensor at a time.  This is deliberately the
    # dtype-conversion point; no converted whole-model copy is ever made.
    # ComfyUI may execute loader nodes under inference_mode().  Tensors
    # allocated there are immutable "inference tensors" outside that context,
    # but our worker refills these slots later on another thread.  Keep the
    # actual in-place copy in a normal-tensor context too.
    #
    # It is also the *dequantization* point for the scale-multiply formats.
    # PyTorch's MPS backend has no Float8 scalar type at all --
    # scalarToMetalTypeString has no case for it, so even `x.to(bfloat16)`
    # fails with "Undefined type Float8_e4m3fn" -- and supports_fp8_compute is
    # CUDA-only, so there is no fp8 matmul on Metal to reach even if the cast
    # worked.  int8 has the same conclusion by a different route: Metal has the
    # type, but comfy_kitchen's mm_int8 goes through torch.int8_mm /
    # torch._int_mm, which is a cuBLASLt path with no MPS implementation, so an
    # int8 weight on this device is dequantized before every use anyway.
    #
    # The CPU backend implements both casts, and the file bytes are already on
    # the CPU here, so converting before the blit is both the only place it can
    # happen and the cheapest: the read stays 1-byte-per-value (half the bytes
    # off disk) while the slot stays a normal compute dtype.
    with torch.inference_mode(False), torch.no_grad():
        if scale is not None or (source.dtype in _QUANTIZED_STORAGE and destination.dtype not in _QUANTIZED_STORAGE):
            # to(destination.dtype) first, then scale, rather than via fp32:
            # fp8_e4m3 carries three mantissa bits and int8 needs seven, both
            # inside bf16's eight, so the widening itself is exact and this
            # avoids a temporary four times the tensor's size.
            converted = source.to(destination.dtype)
            if scale is not None and scale != 1.0:
                # See _cpu_dequantize: to() is a no-op when the dtype matches,
                # and the source may be an inference tensor we must not mutate.
                converted = converted.mul(scale) if converted is source else converted.mul_(scale)
            source = converted
        # copy_ performs the device/dtype conversion into the preallocated slot.
        # Do not spell this as source.to(...): that would create a second,
        # chunk-sized device tensor and defeats the fixed-slot invariant.
        destination.copy_(source)


_SAFETENSORS_DTYPES = {
    "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
    "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
    "U8": torch.uint8, "BOOL": torch.bool,
}
for _name, _attr in (("F8_E4M3", "float8_e4m3fn"), ("F8_E5M2", "float8_e5m2")):
    if hasattr(torch, _attr):
        _SAFETENSORS_DTYPES[_name] = getattr(torch, _attr)

_FP8_DTYPES = tuple(
    getattr(torch, name) for name in ("float8_e4m3fn", "float8_e5m2") if hasattr(torch, name)
)


# QUANTIZATION -- keep the sidecar-key facts here, beside the layout table.
#
# ComfyUI stores per-layer quantization as extra keys next to the weight:
#
# | vintage | marker key | scale key for `LAYER.weight` |
# | v1 (comfy.utils.detect_layer_quantization) | `LAYER.comfy_quant` | `LAYER.weight_scale` |
# | legacy (comfy.utils.convert_old_quants) | top-level `scaled_fp8` | `LAYER.scale_weight` |
#
# The v1 marker is a uint8 tensor holding JSON: {"format": "...", ...}.  The
# format string, not the storage dtype, is what decides whether a layer can be
# streamed, because comfy.quant_ops.QUANT_ALGOS maps three different formats
# onto torch.int8 and two onto torch.float8_e4m3fn:
#
# | format | storage | dequantize | streamable here |
# | float8_e4m3fn / float8_e5m2 | fp8 | w * weight_scale | yes |
# | int8_tensorwise (no convrot) | int8 | w * weight_scale | yes |
# | int8_tensorwise + convrot | int8 | inverse Hadamard, then scale | no |
# | nvfp4 | uint8, 2 values/byte, group 16 | block scales | no |
# | mxfp8 | fp8, group 32, e8m0 block scales | block scales | no |
# | convrot_w4a4 / asym_w4a8_int8 | int8, 4-bit packed | rotation + unpack | no |
#
# The streamable ones are exactly those whose dequantization is one scalar
# multiply, which is what _dtype_copy_ can fold into the copy it is already
# doing.  The rest either change the tensor's stored shape -- so the slot plan
# itself would be wrong -- or need state this loader does not carry.
#
# Neither vintage is loaded here the way ComfyUI loads it.  ComfyUI would build
# MixedPrecisionOps.Linear layers holding QuantizedTensor weights; those are
# unusable on Metal, and they are also unpageable, because
# MixedPrecisionOps.Linear.__init__ never registers `weight` at all (it is
# created inside _load_from_state_dict, which a meta-device build never calls),
# so a slot spec taken from the module would omit every block weight.  So the
# quant config is dropped before the model is built and the scale is folded in
# during the copy instead -- see _dtype_copy_.
_QUANT_MARKER_SUFFIX = ".comfy_quant"
_SCALE_MULTIPLY_FORMATS = ("float8_e4m3fn", "float8_e5m2", "int8_tensorwise")
_QUANTIZED_STORAGE = _FP8_DTYPES + (torch.int8,)


def _scale_key_candidates(tensor_key):
    """Scale keys that could belong to this tensor, most specific first.

    The legacy candidate is deliberately restricted to `.weight`.  Its name is
    parented on the *layer* (`LAYER.scale_weight`), not on the tensor, so an
    unrestricted lookup matches every sibling under that layer -- notably
    `LAYER.bias`, which would then be multiplied by the weight's scale.  Only
    the weight is quantized; the bias is stored at full precision.
    """
    candidates = [tensor_key + "_scale"]
    parent, _, leaf = tensor_key.rpartition(".")
    if parent and leaf == "weight":
        candidates.append(parent + ".scale_weight")
    return tuple(candidates)


def _quantization_scale(reader, tensor_key, cache):
    """Per-tensor dequantization scale for a checkpoint key, or None.

    Cached because every paged block asks for its own keys once per chunk, on
    the execution thread, and a miss costs a dict walk over the whole header.
    """
    if tensor_key in cache:
        return cache[tensor_key]
    scale = None
    for candidate in _scale_key_candidates(tensor_key):
        if candidate not in reader.keys():
            continue
        value = reader.get_tensor(candidate)
        if value.numel() != 1:
            raise RuntimeError(
                f"Lazy DiT: {candidate} holds {value.numel()} values; only per-tensor "
                "(scalar) quantization is supported."
            )
        scale = float(value.reshape(()).to(torch.float32))
        break
    cache[tensor_key] = scale
    return scale


def _layer_quant_format(reader, layer):
    """Parse one `.comfy_quant` blob; None for the legacy scaled_fp8 vintage."""
    marker = layer + _QUANT_MARKER_SUFFIX
    if marker not in reader.keys():
        return None
    try:
        conf = json.loads(reader.get_tensor(marker).numpy().tobytes())
    except Exception as error:
        raise RuntimeError(f"Lazy DiT: could not parse {marker}: {error}") from error
    if not isinstance(conf, dict):
        raise RuntimeError(f"Lazy DiT: {marker} is not a quantization config object.")
    return conf


def _audit_quantization(reader):
    """Reject quantization formats the copy-time dequantizer cannot express.

    Decided from the format string in the marker rather than from the stored
    dtype: int8 alone does not say whether a layer is a plain scale multiply
    or a rotated 4-bit packing.  See the table above.  Refusing here, off the
    header, beats failing at the first copy_ with a shape mismatch.
    """
    markers = [k[:-len(_QUANT_MARKER_SUFFIX)] for k in reader.keys() if k.endswith(_QUANT_MARKER_SUFFIX)]
    quantized = markers
    if not quantized:
        if "scaled_fp8" not in reader.keys():
            return 0
        # Legacy: no per-layer marker, the scale keys are the only evidence,
        # and convert_old_quants writes format "float8_e4m3fn" for all of them.
        quantized = [k[:-len(".scale_weight")] for k in reader.keys() if k.endswith(".scale_weight")]
    for layer in quantized:
        key = layer + ".weight"
        layout = reader.layout(key) if key in reader.keys() else None
        if layout is None:
            raise RuntimeError(f"Lazy DiT: {key} is marked quantized but has no readable layout.")
        conf = _layer_quant_format(reader, layer) or {}
        fmt = conf.get("format", "float8_e4m3fn")
        params = conf.get("params") if isinstance(conf.get("params"), dict) else {}
        if fmt not in _SCALE_MULTIPLY_FORMATS:
            raise RuntimeError(
                f"Lazy DiT cannot stream quantization format {fmt!r} (layer {layer}). "
                f"Streamable formats are {list(_SCALE_MULTIPLY_FORMATS)}: their dequantization is one "
                "scalar multiply, which is folded into the page-in copy. Block-scaled and rotated "
                "formats (nvfp4, mxfp8, convrot_w4a4, asym_w4a8_int8) are not."
            )
        if conf.get("convrot", params.get("convrot", False)):
            raise RuntimeError(
                f"Lazy DiT cannot stream int8_tensorwise with convrot (layer {layer}): the stored "
                "weight is Hadamard-rotated, so recovering it needs more than a scale multiply."
            )
        if layout[2] not in _QUANTIZED_STORAGE:
            raise RuntimeError(
                f"Lazy DiT: {key} is marked {fmt!r} but stored as {layout[2]}, which is not a "
                "storage dtype that format uses. The checkpoint's markers and tensors disagree."
            )
        found = [c for c in _scale_key_candidates(key) if c in reader.keys()]
        if not found:
            raise RuntimeError(f"Lazy DiT: {key} is quantized but the checkpoint has no scale for it.")
        shape = reader.layout(found[0])
        if shape is not None and math.prod(shape[3]) != 1:
            raise RuntimeError(
                f"Lazy DiT: {found[0]} holds {math.prod(shape[3])} values; only per-tensor "
                "(scalar) quantization is supported."
            )
    return len(quantized)


class _DirectReader:
    """Reads streamed block tensors with pread into one reusable buffer.

    safetensors hands back tensors that point into its mmap, so reading a
    tensor through it faults those file pages in and the mapping keeps them
    resident for the life of the handle.  Every forward pass touches every
    block, so a 26 GB checkpoint becomes ~23 GB of resident mapped pages that
    nothing releases -- which is what exhausted the machine.

    This path never maps the file.  It preads into a fixed buffer sized to the
    largest streamed tensor, and sets F_NOCACHE so the kernel does not retain
    the pages either.  Peak stays at the slots plus this one buffer.
    """
    F_NOCACHE = 48    # <sys/fcntl.h>, macOS only

    def __init__(self, path, capacity):
        self.fd = os.open(path, os.O_RDONLY)
        self.nocache = False
        try:
            fcntl.fcntl(self.fd, self.F_NOCACHE, 1)
            self.nocache = True
        except (OSError, AttributeError, ValueError):
            pass       # Linux and older macOS: falls back to normal caching
        self.buffer = bytearray(capacity)
        self.view = memoryview(self.buffer)
        self.capacity = capacity

    def read(self, offset, length, dtype, shape):
        """A CPU tensor over the shared buffer. Valid until the next read()."""
        if length > self.capacity:
            raise RuntimeError(f"streamed tensor of {length} bytes exceeds buffer {self.capacity}")
        target = self.view[:length]
        position, done = offset, 0
        while done < length:
            if _PREADV:
                got = os.preadv(self.fd, [target[done:]], position)
            else:
                block = os.pread(self.fd, length - done, position)
                got = len(block)
                target[done:done + got] = block
            if got <= 0:
                raise RuntimeError(f"short read at {position}: wanted {length - done} more bytes")
            done += got
            position += got
        return torch.frombuffer(target, dtype=dtype).view(shape)

    def close(self):
        try:
            os.close(self.fd)
        except OSError:
            pass


class _PageWarmer:
    """Faults a checkpoint's byte ranges into the page cache. Allocates nothing.

    This is all a worker thread is allowed to do: no torch tensor is created,
    no Metal command is encoded.  Reads go through a single reusable buffer via
    os.preadv (falling back to os.pread), both of which release the GIL for the
    duration of the syscall, so the main thread keeps running.  The bytes read
    are discarded -- the point is the kernel's unified buffer cache, which the
    main thread's later safetensors mmap access then hits instead of disk.

    Cached pages are reclaimable under pressure rather than an allocation, so
    the resident-slot budget still means what it says.
    """
    SLICE = 4 << 20

    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY)
        self.view = memoryview(bytearray(self.SLICE))

    def warm(self, ranges):
        read = 0
        for start, length in _coalesce(ranges):
            offset, end = start, start + length
            while offset < end:
                want = min(self.SLICE, end - offset)
                if _PREADV:
                    got = os.preadv(self.fd, [self.view[:want]], offset)
                else:
                    got = len(os.pread(self.fd, want, offset))
                if got <= 0:
                    break
                offset += got
                read += got
        return read

    def close(self):
        try:
            os.close(self.fd)
        except OSError:
            pass


def _coalesce(ranges):
    """Merge touching/adjacent byte ranges so warming issues few large reads."""
    merged = []
    for start, length in sorted(ranges):
        if merged and start <= merged[-1][0] + merged[-1][1]:
            previous_start, previous_length = merged[-1]
            merged[-1] = (previous_start, max(previous_length, start + length - previous_start))
        else:
            merged.append((start, length))
    return merged


@dataclasses.dataclass
class _Job:
    chunk: int
    ready: threading.Event = dataclasses.field(default_factory=threading.Event)
    error: str | None = None
    seconds: float = 0.0


class _SlotStore(torch.nn.Module):
    """Registers slot tensors so PyTorch/Comfy memory accounting can see them."""
    def __init__(self):
        super().__init__()
        self.values = {}
    def add(self, key, tensor, is_param):
        registered = f"slot_{len(self.values)}"
        if is_param:
            self.register_parameter(registered, tensor)
        else:
            self.register_buffer(registered, tensor, persistent=False)
        self.values[key] = tensor


class _BlockSlots:
    """Reusable parameter sets for one homogeneous ModuleList kind."""
    def __init__(self, modules, device, slot_positions, store):
        self.modules, self.device = modules, device
        exemplar = modules[0]
        self.spec = [(n, is_param, tuple(t.shape), t.dtype) for n, is_param, t in _named_local_tensors(exemplar)]
        signature = [(n, is_param, shape, dtype) for n, is_param, shape, dtype in self.spec]
        for index, module in enumerate(modules[1:], 1):
            other = [(n, is_param, tuple(t.shape), t.dtype) for n, is_param, t in _named_local_tensors(module)]
            if other != signature:
                raise RuntimeError(
                    "Lazy DiT discovered a ModuleList with heterogeneous parameter geometry "
                    f"at block {index}. Add an explicit shape-kind override before streaming it."
                )
        self.slots = {}
        self.host = {}
        for section, local_index in sorted(slot_positions):
            d = {}
            for name, is_param, shape, dtype in self.spec:
                # Refilled on the execution thread at each chunk boundary.
                # Never allocate them as inference tensors.
                with torch.inference_mode(False):
                    if is_param:
                        tensor = torch.nn.Parameter(torch.empty(shape, dtype=dtype, device=device), requires_grad=False)
                    else:
                        tensor = torch.empty(shape, dtype=dtype, device=device)
                store.add((id(self), section, local_index, name), tensor, is_param)
                d[name] = tensor
                # Cached once: slots are allocated at load and never freed, so
                # the buffer cannot be recycled underneath us.
                self.host[(section, local_index, name)] = _host_pointer(tensor)
            self.slots[(section, local_index)] = d
    @property
    def bytes_per_slot(self):
        return sum(t.numel() * t.element_size() for t in next(iter(self.slots.values())).values())
    def bind_chunk_block(self, module, section, local_index):
        """Install this slot's storage on `module`, as tensor objects unique to this bind.

        The storage is deliberately shared: slot (section, local) serves every
        block whose index is congruent, so blocks b and b+2c alternate through
        the same bytes.  Handing out the *same Python object* each time made
        that invisible to consumers -- a downstream cache of derived weight
        state (an fp8 cast, a compiled kernel, a Metal buffer handle) keyed on
        tensor identity would see one unchanging weight and reuse a conversion
        of some other block's numbers.  A fresh object per bind makes every
        identity- or weakref-keyed cache miss and recompute, which is the
        truthful answer: these really are different weights now.

        A cache keyed on data_ptr() or storage cannot be corrected this way --
        see the aliasing note in the module docstring.
        """
        slot = self.slots[(section, local_index)]
        for name, is_param, _, _ in self.spec:
            # detach() yields a new tensor object over the same storage; no
            # weight bytes are copied.  Not an inference tensor: the bound
            # object outlives this call and is read during ordinary execution.
            with torch.inference_mode(False):
                view = slot[name].detach()
                bound = torch.nn.Parameter(view, requires_grad=False) if is_param else view
            _set_qualified(module, name, bound, buffer=not is_param)
        return slot


class LazyBlockPager:
    """The specified two-section schedule; chunk boundaries are flat/global."""
    WAIT_SECONDS = 300.0
    def __init__(self, root, reader, groups, k, debug=False, checkpoint_prefix="", key_transform=_checkpoint_key, prefetch=True,
                 owner=None, patch_prefix=""):
        self.root, self.reader, self.groups, self.k, self.debug = root, reader, groups, k, debug
        self.key_transform = key_transform
        # WEIGHT PATCHES (LoRA and friends).  A paged block's storage is
        # overwritten from the checkpoint at every chunk boundary, so a patch
        # merged into it once would survive exactly until the slot's next
        # refill.  Patches on paged tensors are therefore reapplied inside
        # _copy_chunk, immediately after the copy that would have erased them.
        #
        # `owner` is the BaseModel; ModelPatcher.pre_run sets its
        # `current_patcher`, so reading the patches from there at forward entry
        # uses whichever patcher is actually sampling.  Clones share one model
        # and thus one pager, so a workflow with two different LoRA stacks over
        # the same checkpoint would otherwise get whichever clone was loaded
        # last.  `patch_prefix` maps a block's model-side name into the
        # patcher's key space ("diffusion_model." for a DiT; a text encoder
        # pager passes its own).
        # Every slot in every group lives on one device; take it from the
        # first _BlockSlots rather than re-deriving it.
        self.device = next(iter(groups.values()))[1].device
        self.owner = owner
        self.patch_prefix = patch_prefix
        self.patches = {}
        self.pinned_patches = {}
        self.block_prefixes = tuple(patch_prefix + group[3] + "." for group in groups.values())
        # prefetch=False skips page warming entirely, so no thread but the
        # caller's runs.  Correctness does not depend on it either way -- the
        # copies are on the execution thread in both modes -- so this is a
        # speed switch and a bisection tool, not a safety valve.
        self.prefetch = prefetch
        self.blocks = [(kind, i, module) for kind, group in groups.items() for i, module in enumerate(group[0])]
        self.n, self.c = len(self.blocks), math.ceil(k / 2)
        self.chunks = [(i, min(i + self.c, self.n)) for i in range(0, self.n, self.c)]
        self.lock = threading.RLock()
        self.jobs = {}
        self.checkpoint_prefix = checkpoint_prefix
        # Persists across forward passes: the file's scales never change, and
        # _reset must not throw the lookup work away every step.
        self.scales = {}
        self.timing = {"warm_seconds": {}, "wait_seconds": {}, "copy_seconds": {},
                       "read_seconds": {}, "write_seconds": {}, "drain_seconds": {},
                       "fence_seconds": {}, "bytes": {}}
        self._logged_dtypes = False
        # After checkpoint_prefix and chunks: _open_direct walks _chunk_keys.
        self.direct = self._open_direct(reader)
        self.direct_worker = self._open_direct(reader) if self.direct is not None else None
        self.warmer = _PageWarmer(reader.path) if (prefetch and getattr(reader, "path", None)) else None
        self.event_class = getattr(getattr(torch, "mps", None), "Event", None)
        self.async_capable = self._check_async_capable()
        self.async_refill = False       # per-pass; _reset decides
        self._install_hooks()

    def _check_async_capable(self):
        """Can refills run off the Metal queue, on a worker thread?

        Requires a host-visible address for every slot tensor (see
        _host_pointer), an mps.Event to fence the section handover, its own
        reader, and more than one chunk -- with a single chunk there is nothing
        to overlap with.
        """
        if self.event_class is None:
            logging.info("Lazy DiT: no torch.mps.Event; refills stay on the queue")
            return False
        if self.direct_worker is None or len(self.chunks) < 2 or not self.prefetch:
            logging.info("Lazy DiT: refills stay on the queue (direct_reader=%s chunks=%d prefetch=%s)",
                         self.direct_worker is not None, len(self.chunks), self.prefetch)
            return False

        # Collect every slot's address and extent, then prove no two overlap.
        # This is the real safety property: if the allocator ever handed two
        # slots regions of one buffer, a memmove into one would silently
        # corrupt the other, and no size check on an individual tensor can
        # detect that.  Comparing the addresses can.
        extents = []
        for group in self.groups.values():
            slots = group[1]
            if not slots.host:
                logging.info("Lazy DiT: no slot tensors to address; refills stay on the queue")
                return False
            for key, address in slots.host.items():
                if address is None:
                    section, local_index, name = key
                    logging.info(
                        "Lazy DiT: slot %s (section %d, %d) is not host-addressable; refills stay "
                        "on the queue", name, section, local_index)
                    return False
                tensor = slots.slots[(key[0], key[1])][key[2]]
                extents.append((address, tensor.numel() * tensor.element_size(), key))
        extents.sort()
        for (start, length, key), (next_start, _, next_key) in zip(extents, extents[1:]):
            if start + length > next_start:
                logging.warning(
                    "Lazy DiT: slots %s and %s share storage (%#x+%d overlaps %#x); refills stay "
                    "on the queue", key[2], next_key[2], start, length, next_start)
                return False
        logging.info("Lazy DiT: off-queue refills available (%d slot tensors, no overlap)", len(extents))
        return True
    def _open_direct(self, reader):
        """Direct reader sized to the largest streamed tensor, if we can size it."""
        if not getattr(reader, "path", None) or not hasattr(reader, "layout"):
            return None
        capacity = 0
        for j in range(len(self.chunks)):
            for _, _, key, _, _ in self._chunk_keys(j):
                layout = reader.layout(key) if key in reader.keys() else None
                if layout is None:
                    logging.info("Lazy DiT: %s has no direct layout; falling back to mmap reads", key)
                    return None
                capacity = max(capacity, layout[1])
        if capacity == 0:
            return None
        try:
            direct = _DirectReader(reader.path, capacity)
        except Exception:
            logging.info("Lazy DiT: direct reads unavailable; falling back to mmap reads", exc_info=True)
            return None
        logging.info("Lazy DiT: direct reads, %.1f MiB buffer, F_NOCACHE=%s", capacity / 2**20, direct.nocache)
        return direct
    def _install_hooks(self):
        # A pre-hook precedes the actual block operation, including Comfy patch replacement.
        for absolute, (_, _, module) in enumerate(self.blocks):
            module.register_forward_pre_hook(lambda _m, _a, index=absolute: self._enter(index), with_kwargs=False)
        original = self.root.forward
        def wrapped(this, *args, **kwargs):
            with self.lock:              # Comfy may issue concurrent CFG forwards.
                self._reset()
                return original(*args, **kwargs)
        self.root.forward = types.MethodType(wrapped, self.root)
    def close(self):
        for handle in (self.direct, self.direct_worker, self.warmer):
            if handle is not None:
                handle.close()
        self.direct = self.direct_worker = self.warmer = None
    def _reset(self):
        self.jobs = {}
        self._refresh_patches()
        # Patches no longer force this off: _cpu_merge_patches applies them to
        # the CPU tensor before the memmove, so the worker still issues no
        # Metal command.  The queued fallback keeps the device-side
        # _apply_patches for the case where a slot has no host address.
        self.async_refill = self.async_capable
        self.timing = {"warm_seconds": {}, "wait_seconds": {}, "copy_seconds": {},
                       "read_seconds": {}, "write_seconds": {}, "drain_seconds": {},
                       "fence_seconds": {}, "bytes": {}}

    def _refresh_patches(self):
        """Adopt the weight patches that apply to this forward pass.

        Called under self.lock at forward entry, so the dict cannot change
        while a pass is copying chunks out of it.

        Two sources, in order.  A BaseModel gets `current_patcher` set by
        ModelPatcher.pre_run, which is the reliable answer for a DiT: clones
        share one model and therefore one pager, so a workflow with two LoRA
        stacks over one checkpoint must be resolved per pass, not per load.
        A text encoder wrapper (sd1_clip.SD1ClipModel) has no such attribute --
        nothing calls pre_run on a CLIP patcher -- so there the pager uses
        whatever patch_model last pinned.
        """
        patcher = getattr(self.owner, "current_patcher", None) if self.owner is not None else None
        patches = getattr(patcher, "patches", None)
        if not patches:
            patches = self.pinned_patches
        self.patches = patches if patches else {}

    def _cpu_merge_patches(self, source, key):
        """Apply `key`'s patches to a CPU tensor before it is memmoved into a slot.

        Same arithmetic as _apply_patches, on the CPU.  This is what lets a
        LoRA keep the off-queue refill: the GPU version has to run Metal
        commands, which a worker thread must not do, so before this existed any
        patch dropped the whole pass back to synchronous refills -- and a
        4-step Lightning LoRA touches nearly every paged tensor, so that was
        the common case, not the corner.

        The rank-r matmul is far cheaper than the bytes already being moved,
        and it happens on the worker while the GPU computes, so it costs
        nothing visible.
        """
        patches = self.patches.get(key)
        if not patches:
            return source
        with torch.inference_mode(False), torch.no_grad():
            host = torch.device("cpu")
            compute_dtype = _lora_compute_dtype(host)
            temporary = source.to(dtype=compute_dtype, copy=True)
            merged = comfy.lora.calculate_weight(patches, temporary, key)
            if merged.dtype != source.dtype:
                merged = comfy.float.stochastic_rounding(merged, source.dtype, seed=comfy.utils.string_to_seed(key))
        return merged

    def _apply_patches(self, destination, key):
        """Merge `key`'s patches into a slot tensor that was just refilled.

        Mirrors ModelPatcher.patch_weight_to_device's arithmetic -- compute in
        the lora dtype, stochastically round back to the slot dtype -- but
        writes in place into the existing slot rather than replacing the
        module's parameter, because the slot's address is the whole point.  No
        backup is kept: the next refill restores the checkpoint value for free,
        which is also why toggling a LoRA needs no unpatch pass here.
        """
        patches = self.patches.get(key)
        if not patches:
            return
        with torch.inference_mode(False), torch.no_grad():
            compute_dtype = _lora_compute_dtype(destination.device)
            temporary = comfy.model_management.cast_to_device(destination, destination.device, compute_dtype, copy=True)
            merged = comfy.lora.calculate_weight(patches, temporary, key)
            if merged.dtype != destination.dtype:
                merged = comfy.float.stochastic_rounding(merged, destination.dtype, seed=comfy.utils.string_to_seed(key))
            destination.copy_(merged)
        del temporary, merged
    def _chunk_for(self, index): return index // self.c
    def _enter(self, index):
        if index % self.c:
            return
        j = self._chunk_for(index)
        started = time.perf_counter()
        if self.async_refill:
            # The worker already wrote this chunk's slot storage straight into
            # the Metal buffers, off the queue and while the GPU was busy.  All
            # that is left on this thread is binding, which touches no bytes.
            job = self.jobs.get(j)
            if job is None:                       # chunk 0, or a worker refused
                self._fill_chunk(j)
            else:
                if not job.ready.wait(self.WAIT_SECONDS):
                    raise TimeoutError(f"Lazy DiT timed out waiting {self.WAIT_SECONDS}s for chunk {j}")
                self.timing["wait_seconds"][j] = time.perf_counter() - started
                if job.error:
                    raise RuntimeError(f"Lazy DiT worker failed on chunk {j}:\n{job.error}")
            self._bind_chunk(j)
            # Recorded now, not when the worker starts: everything submitted so
            # far includes chunk j-1's compute, and chunk j+1 reuses exactly
            # chunk j-1's section.  Chunk j's own compute is submitted after
            # this returns and is irrelevant to that section.
            self._spawn(j + 1, fence=self._record_fence())
        else:
            if self.prefetch:
                job = self.jobs.get(j)   # absent for chunk 0, and after a warm failure
                if job is not None:
                    if not job.ready.wait(self.WAIT_SECONDS):
                        raise TimeoutError(f"Lazy DiT timed out waiting {self.WAIT_SECONDS}s for chunk {j}")
                    self.timing["wait_seconds"][j] = time.perf_counter() - started
                    if job.error and self.debug:
                        # Warming is an optimization; a failed warm costs speed,
                        # not correctness -- the copy below reads the file anyway.
                        logging.info("Lazy DiT warm for chunk %d failed:\n%s", j, job.error)
                self._spawn(j + 1)       # read ahead while this chunk is copied in
            self._fill_chunk(j, bind=True)
        self.timing["copy_seconds"][j] = time.perf_counter() - started
        if self.debug:
            mib = self.timing["bytes"].get(j, 0) / 2**20
            warm = self.timing["warm_seconds"].get(j, 0.0)
            read = self.timing["read_seconds"].get(j, 0.0)
            write = self.timing["write_seconds"].get(j, 0.0)
            drain = self.timing["drain_seconds"].get(j, 0.0)
            fence = self.timing["fence_seconds"].get(j, 0.0)
            logging.info(
                "Lazy DiT chunk %d: %.0f MiB | warm=%.3fs (%.1f GiB/s) wait=%.3fs drain=%.3fs "
                "fence=%.3fs copy=%.3fs [read=%.3fs (%.1f GiB/s) write=%.3fs (%.1f GiB/s)]%s",
                j, mib, warm, (mib / 1024) / warm if warm else 0.0,
                self.timing["wait_seconds"].get(j, 0.0), drain, fence, self.timing["copy_seconds"][j],
                read, (mib / 1024) / read if read else 0.0,
                write, (mib / 1024) / write if write else 0.0,
                " (async)" if self.async_refill else ("" if self.prefetch else " (synchronous)"))

    def _record_fence(self):
        """Event marking every command submitted so far. None if unavailable."""
        if self.event_class is None:
            return None
        try:
            fence = self.event_class()
            fence.record()
            return fence
        except Exception:
            logging.info("Lazy DiT: mps.Event unusable; refills stay synchronous", exc_info=True)
            self.async_refill = False
            return None

    def _spawn(self, j, fence=None):
        if j >= len(self.chunks) or j in self.jobs:
            return
        if not self.async_refill:
            if self.warmer is None:
                return
            job = self.jobs[j] = _Job(j)
            ranges = self._chunk_ranges(j)
            def warm_only():
                try:
                    started = time.perf_counter()
                    self.warmer.warm(ranges)
                    job.seconds = time.perf_counter() - started
                    self.timing["warm_seconds"][j] = job.seconds
                except Exception:
                    job.error = traceback.format_exc()
                finally:
                    job.ready.set()
            threading.Thread(target=warm_only, name=f"lazy-dit-warm-{j}", daemon=True).start()
            return

        job = self.jobs[j] = _Job(j)
        def fill():
            try:
                # The slots about to be overwritten were last read by chunk
                # j-2's blocks.  The fence was recorded after those were
                # submitted, so waiting on it is exactly the guarantee the
                # Metal queue used to give for free -- and no more: it does not
                # wait for the compute currently running.
                if fence is not None:
                    started = time.perf_counter()
                    fence.synchronize()
                    self.timing["fence_seconds"][j] = time.perf_counter() - started
                self._fill_chunk(j, worker=True)
            except Exception:
                job.error = traceback.format_exc()
            finally:
                job.ready.set()
        threading.Thread(target=fill, name=f"lazy-dit-fill-{j}", daemon=True).start()
    def _chunk_keys(self, j):
        """Yield (kind, absolute index, checkpoint key, local name, patch key).

        The last two are the same tensor named two ways.  `local_name` is the
        module-side name used to index the bound slot; `patch_key` is that name
        qualified up to the patcher's root, which is the key space ModelPatcher
        stores LoRA patches under.  The checkpoint key is neither: it has been
        through key_transform (Flux renames *_norm.weight to *_norm.scale).
        """
        start, end = self.chunks[j]
        for absolute in range(start, end):
            kind, block_index, _ = self.blocks[absolute]
            prefix = self.checkpoint_prefix + self.groups[kind][2] + f"{self.groups[kind][3]}.{block_index}."
            patch_prefix = self.patch_prefix + f"{self.groups[kind][3]}.{block_index}."
            for local_name, _, _, _ in self.groups[kind][1].spec:
                yield kind, absolute, self.key_transform(prefix + local_name), local_name, patch_prefix + local_name
    def _chunk_ranges(self, j):
        ranges = []
        for _, _, key, _, _ in self._chunk_keys(j):
            span = self.reader.byte_range(key)
            if span is not None:
                ranges.append(span)
        return ranges
    def _bind_chunk(self, j):
        """Install this chunk's slot storage on its modules. Execution thread.

        Separate from filling because binding mutates modules and creates
        tensor objects, neither of which a worker thread may do -- while
        filling, after this split, touches nothing but raw addresses.
        """
        start, _ = self.chunks[j]
        section = j % 2
        for absolute in range(start, self.chunks[j][1]):
            kind, _, module = self.blocks[absolute]
            self.groups[kind][1].bind_chunk_block(module, section, absolute - start)

    def _fill_chunk(self, j, bind=False, worker=False):
        """Read one chunk and write it into its slot storage.

        Runs on a worker thread when async_refill is on.  Everything it does is
        then a file read plus a memmove into an address: no torch device op, no
        Metal command, no module mutation.  That is what lets it overlap the
        compute the GPU is already running, which is the whole point -- the
        queued copy_ it replaces could not start until that compute drained.

        `bind` is for the synchronous path, where binding and filling are the
        same pass.  `worker` selects the second reader, since _DirectReader
        hands out views over one reusable buffer and two threads cannot share
        it.
        """
        start, end = self.chunks[j]
        section = j % 2
        reader = self.direct_worker if worker else self.direct
        bound = {}
        if not worker:
            # A queued copy_ blocks until the queue reaches it, and at a chunk
            # boundary the queue still holds the compute for the c blocks that
            # just ran.  Without this, that compute is billed to whichever
            # copy_ happens to be first and shows up as an impossibly slow
            # "write".  Draining costs nothing extra -- the wait happens either
            # way -- and makes the two numbers mean what they say.  The worker
            # path never drains: its fence already covers the only ordering
            # that matters, and a full drain there would give back exactly the
            # overlap it exists to win.
            mark = time.perf_counter()
            _synchronize_device(self.device)
            self.timing["drain_seconds"][j] = time.perf_counter() - mark
        read_seconds = write_seconds = 0.0
        read_bytes = 0
        dtype_bytes = {}
        for kind, absolute, key, local_name, patch_key in self._chunk_keys(j):
            slots = self.groups[kind][1]
            local_index = absolute - start
            if bind and absolute not in bound:
                _, _, module = self.blocks[absolute]
                bound[absolute] = slots.bind_chunk_block(module, section, local_index)
            destination = slots.slots[(section, local_index)][local_name]
            if key not in self.reader.keys():
                raise KeyError(f"checkpoint lacks streamed tensor {key}")
            mark = time.perf_counter()
            if reader is not None:
                offset, length, dtype, shape = self.reader.layout(key)
                source = reader.read(offset, length, dtype, shape)
            else:
                source = self.reader.get_tensor(key)
            read_seconds += time.perf_counter() - mark
            pair = (source.dtype, destination.dtype)
            dtype_bytes[pair] = dtype_bytes.get(pair, 0) + source.numel() * source.element_size()
            read_bytes += source.numel() * source.element_size()

            mark = time.perf_counter()
            scale = _quantization_scale(self.reader, key, self.scales)
            address = slots.host.get((section, local_index, local_name))
            if address is not None:
                # Off-queue write.  Conversion and patching both happen on the
                # CPU first -- there is no such thing as a converting or
                # patching memmove -- and the slot is written once, already
                # correct.  A patched tensor therefore never exists in a
                # half-refilled state at the slot address.
                converted = _cpu_dequantize(source, destination.dtype, scale)
                converted = self._cpu_merge_patches(converted, patch_key)
                _host_write(address, converted)
                del converted
            else:
                _dtype_copy_(destination, source, scale)
                if self.patches:
                    # Immediately, before anything can read the slot: the copy
                    # above just replaced whatever this address held with raw
                    # checkpoint bytes, so this is the only moment at which a
                    # patch can be merged and still be the value the block
                    # computes with.
                    self._apply_patches(destination, patch_key)
            write_seconds += time.perf_counter() - mark
            del source
        self.timing["read_seconds"][j] = read_seconds
        self.timing["write_seconds"][j] = write_seconds
        self.timing["bytes"][j] = read_bytes
        if not self._logged_dtypes:
            # Report the whole chunk by byte volume, not whichever tensor came
            # first: a checkpoint mixes dtypes, and the small F32 norm scales
            # are not what the copy time is spent on.
            self._logged_dtypes = True
            for (file_dtype, slot_dtype), count in sorted(dtype_bytes.items(), key=lambda item: -item[1]):
                logging.info(
                    "Lazy DiT copy: %7.1f MiB  %s -> %s  (%s)", count / 2**20, file_dtype, slot_dtype,
                    "blit" if file_dtype == slot_dtype else "converting",
                )


def _lora_compute_dtype(device):
    getter = getattr(comfy.model_management, "lora_compute_dtype", None)
    return getter(device) if getter is not None else torch.float32


class LazyDiTModelPatcher(comfy.model_patcher.ModelPatcher):
    """Keeps Comfy's offloader off the pager slots without disabling patching.

    The signature is the base class's, not a convenience one.  ModelPatcher
    .clone() rebuilds through ``self.__class__(model, load_device,
    offload_device, self.model_size(), weight_inplace_update=...)``, so any
    subclass that narrows __init__ breaks every node that clones -- LoRA
    loaders, ModelSampling*, FluxKVCache.  ``disable_dynamic`` is accepted and
    ignored for the same reason: clone(force_deepcopy=True) re-invokes
    cached_patcher_init with it.

    What is genuinely overridden is only the weight *movement*: load,
    partially_load and partially_unload, because the pager owns those bytes and
    a second mover would double-count them or hand a slot to the offloader.
    Object patches, injections and hooks are left to run normally, and weight
    patches are split by destination: resident tensors are patched here through
    the stock machinery, paged ones are handed to the pager, which reapplies
    them at each refill.
    """
    def __init__(self, model, load_device, offload_device=None, size=0,
                 weight_inplace_update=False, disable_dynamic=False):
        super().__init__(model, load_device,
                         load_device if offload_device is None else offload_device,
                         size=size, weight_inplace_update=weight_inplace_update)

    @property
    def pager(self):
        # Set by whichever loader built this patcher's model.  Not
        # `self.model.diffusion_model._lazy_dit_pager`: the same patcher class
        # backs the text encoder, whose model is an sd1_clip wrapper with no
        # diffusion_model at all.
        return getattr(self.model, "_lazy_pager", None)

    def model_size(self): return self.size
    def loaded_size(self): return self.size
    def load(self, *args, **kwargs): return self

    def _resident_patch_keys(self):
        """Patched keys that are NOT inside a paged block list.

        Flux LoRAs routinely touch img_in, time_in, vector_in and final_layer,
        which _load_nonblocks materialized once as ordinary parameters.  Those
        are patched exactly like an unpaged model -- with a backup, so
        unpatch_model can put them back.
        """
        pager = self.pager
        prefixes = pager.block_prefixes if pager is not None else ()
        return [k for k in self.patches if not any(k.startswith(p) for p in prefixes)]

    def patch_model(self, device_to=None, lowvram_model_memory=0, load_weights=True, force_patch_weights=False):
        with self.use_ejected():
            for key in self.object_patches:
                old = comfy.utils.set_attr(self.model, key, self.object_patches[key])
                if key not in self.object_patches_backup:
                    self.object_patches_backup[key] = old
            for key in self._resident_patch_keys():
                self.patch_weight_to_device(key, device_to=self.load_device)
            pager = self.pager
            if pager is not None:
                pager.pinned_patches = self.patches
            self.model.current_weight_patches_uuid = self.patches_uuid
        self.inject_model()
        return self.model

    def unpatch_model(self, device_to=None, unpatch_weights=True):
        self.eject_model()
        if unpatch_weights:
            self.unpatch_hooks()
            for key in list(self.backup.keys()):
                backup = self.backup[key]
                if backup.inplace_update:
                    comfy.utils.copy_to_param(self.model, key, backup.weight)
                else:
                    comfy.utils.set_attr_param(self.model, key, backup.weight)
            self.backup.clear()
            pager = self.pager
            if pager is not None:
                pager.pinned_patches = {}
            self.model.current_weight_patches_uuid = None
        for key in list(self.object_patches_backup.keys()):
            comfy.utils.set_attr(self.model, key, self.object_patches_backup[key])
        self.object_patches_backup.clear()
        # device_to is deliberately not honored: self.model.to(device) would
        # walk the slot tensors, and the offload device is the load device here
        # anyway.
        return self.model

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        # The base implementation's shape (unpatch, then repatch without
        # loading weights) is what makes a patches_uuid change take effect;
        # only its call into self.load is dropped.
        with self.use_ejected(skip_and_inject_on_exit_only=True):
            stale = (self.model.current_weight_patches_uuid is not None
                     and (self.model.current_weight_patches_uuid != self.patches_uuid or force_patch_weights))
            self.unpatch_model(self.offload_device, unpatch_weights=stale)
            self.patch_model(load_weights=False)
        return 0

    def partially_unload(self, *args, **kwargs): return 0

    def detach(self, unpatch_all=True):
        self.eject_model()
        if unpatch_all:
            self.unpatch_model(self.offload_device, unpatch_weights=True)
        for callback in self.get_all_callbacks(comfy.model_patcher.CallbacksMP.ON_DETACH):
            callback(self, unpatch_all)
        return self.model


# ARCHITECTURE TABLE -- root class -> (ordered block lists, model->checkpoint
# key rewrite).  A missing entry falls back to autodetected lists and the
# identity rewrite, which is the honest default: Flux is the only
# supported_models config in the audited checkout that overrides
# process_unet_state_dict, so every other target's module names are already
# its checkpoint names.
#
# Getting the rewrite wrong is not a near miss.  _checkpoint_key rewrites any
# `*_norm.weight`, and QwenImageTransformer2DModel has a root-level
# `txt_norm.weight`; applying Flux's rule to it asks the file for
# `txt_norm.scale`, which no Qwen-Image checkpoint has.
_DIT_ARCHITECTURES = {
    "Flux": (("double_blocks", "single_blocks"), _checkpoint_key),
    "SingleStreamDiT": (("blocks",), _identity_key),
    "QwenImageTransformer2DModel": (("transformer_blocks",), _identity_key),
}


def _find_block_lists(root):
    """(block lists, key transform) for this DiT root."""
    cls = root.__class__.__name__
    names, key_transform = _DIT_ARCHITECTURES.get(cls, (None, _identity_key))
    attrs = [(n, v) for n, v in root.named_children() if isinstance(v, torch.nn.ModuleList)]
    if names is None:
        # Generic fallback: module lists whose nonempty entries have one type.
        names = tuple(n for n, v in attrs if len(v) and len({type(x) for x in v}) == 1)
    found = []
    for name in names:
        value = getattr(root, name, None)
        if not isinstance(value, torch.nn.ModuleList) or not len(value):
            raise RuntimeError(f"Lazy DiT does not support {cls}: expected block list {name!r}; ModuleLists found: {[n for n, _ in attrs]}")
        found.append((name, value))
    if not found:
        raise RuntimeError(f"Lazy DiT found no supported transformer blocks on {cls}; ModuleLists found: {[n for n, _ in attrs]}")
    return found, key_transform


def _load_nonblocks(root, reader, block_prefixes, checkpoint_prefix="", key_transform=_checkpoint_key, strict=True, scales=None):
    """Materialize every tensor outside the paged block lists.

    `strict` mirrors the two upstream loaders.  A DiT config is *derived from
    the checkpoint* by comfy.model_detection, so a tensor the module wants and
    the file lacks means detection went wrong -- raise.  A text encoder is
    built from a hardcoded config instead, so optional tensors are normal:
    sd1_clip.load_sd is load_state_dict(strict=False) and sd.py only logs
    missing keys.  Qwen3-8B Klein ties its embeddings and ships no
    lm_head.weight.  Returns the missing model-side names.
    """
    parameters = dict(root.named_parameters(recurse=True))
    buffers = dict(root.named_buffers(recurse=True))
    if scales is None:
        scales = {}
    missing = []
    for name, tensor in list(parameters.items()) + list(buffers.items()):
        if any(name.startswith(p) for p in block_prefixes):
            continue
        key = checkpoint_prefix + key_transform(name)
        if key not in reader.keys():
            if strict:
                raise KeyError(f"checkpoint lacks resident tensor {key}")
            missing.append(key)
            continue
        src = reader.get_tensor(key)
        # This permanent resident tensor must remain writable if Comfy later
        # applies a supported non-streamed patch, so avoid inference tensors.
        with torch.inference_mode(False):
            dst = torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device=root.device)
        _dtype_copy_(dst, src, _quantization_scale(reader, key, scales))
        _set_qualified(root, name, torch.nn.Parameter(dst, requires_grad=False) if name in parameters else dst, buffer=name in buffers)
        del src
    return missing


def _bytes_not_in_block_lists(root, prefixes):
    tensors = list(root.named_parameters(recurse=True)) + list(root.named_buffers(recurse=True))
    return sum(t.numel() * t.element_size() for name, t in tensors if not any(name.startswith(p) for p in prefixes))


def _slot_plan(lists, k):
    """Return flat execution order and exactly the typed slots this k can touch."""
    blocks = [(name, i, module) for name, modules in lists for i, module in enumerate(modules)]
    c = math.ceil(k / 2)
    positions = {name: set() for name, _ in lists}
    for absolute, (kind, _, _) in enumerate(blocks):
        chunk = absolute // c
        positions[kind].add((chunk % 2, absolute % c))
    return blocks, c, positions


def _choose_k_for_budget(root, lists, budget_bytes):
    """Largest non-full k whose exact typed-slot allocation fits budget."""
    prefixes = tuple(name + "." for name, _ in lists)
    resident = _bytes_not_in_block_lists(root, prefixes)
    n = sum(len(modules) for _, modules in lists)
    # n-1 can allocate every physical slot for an odd/even single-list model.
    # Keep at least two logical blocks absent, satisfying the no-full-model rule.
    maximum_k = max(2, n - 2)
    candidates = []
    for k in range(2, maximum_k + 1):
        _, _, positions = _slot_plan(lists, k)
        slot_bytes = 0
        for name, modules in lists:
            one_block = sum(t.numel() * t.element_size() for _, _, t in _named_local_tensors(modules[0]))
            slot_bytes += len(positions[name]) * one_block
        if resident + slot_bytes <= budget_bytes:
            candidates.append((k, slot_bytes))
    if not candidates:
        _, _, positions = _slot_plan(lists, 2)
        minimum_slots = sum(
            len(positions[name]) * sum(t.numel() * t.element_size() for _, _, t in _named_local_tensors(modules[0]))
            for name, modules in lists
        )
        minimum = resident + minimum_slots
        raise RuntimeError(
            f"Lazy DiT memory_limit_mib={budget_bytes / 2**20:.1f} is too small: "
            f"the fixed resident tensors plus the mandatory two slots need at least {minimum / 2**20:.1f} MiB."
        )
    k, slot_bytes = candidates[-1]
    return k, resident, slot_bytes


def load_lazy_diffusion_model(unet_path, model_options, memory_limit_mib, debug, prefetch=True, disable_dynamic=False):
    # disable_dynamic is ignored: this patcher is never dynamic.  It exists
    # because ModelPatcher.clone(force_deepcopy=True) re-invokes
    # cached_patcher_init with that keyword.
    if not unet_path.lower().endswith(".safetensors"):
        raise RuntimeError("Lazy DiT requires a single .safetensors diffusion-model file; it never uses ComfyUI's full-state-dict loader.")
    reader = _TensorReader(unet_path)  # one safe_open handle; kept by pager
    shape_sd = _ShapeOnlyStateDict(reader)
    config = comfy.model_detection.model_config_from_unet(shape_sd, "", metadata=reader.metadata())
    if config is None:
        raise RuntimeError("ComfyUI could not detect this safetensors diffusion model from tensor names/shapes.")
    scales = {}
    quantized_layers = _audit_quantization(reader)
    if getattr(config, "quant_config", None):
        # Detection sets this from the `.comfy_quant` keys and pick_operations
        # turns it into MixedPrecisionOps.  Clearing it gives plain
        # manual_cast Linear layers holding ordinary dense weights, which is
        # both pageable and computable here; the fp8 values are unpacked
        # during the copy instead.  See the quantization table above.
        config.quant_config = None
    # Match comfy.sd.load_diffusion_model_state_dict dtype policy without loading values.
    device = comfy.model_management.get_torch_device()
    dtype = model_options.get("dtype") or comfy.model_management.unet_dtype(
        model_params=-1,
        supported_dtypes=config.supported_inference_dtypes,
    )
    if dtype in _FP8_DTYPES and not comfy.model_management.supports_fp8_compute(device):
        # Slots would be allocated fp8 and every use would have to dequantize.
        # On MPS that dequantize does not exist (see _dtype_copy_), and where
        # it does exist it is strictly slower than just holding the compute
        # dtype.  Fail here rather than several frames into a Metal kernel.
        raise RuntimeError(
            f"Lazy DiT: weight_dtype={dtype} needs fp8 compute, which {device} does not have. "
            "Use fp16, bf16, or default. An fp8 checkpoint still loads under those settings -- "
            "it is unpacked during paging, so the file reads stay fp8-width."
        )
    manual = comfy.model_management.unet_manual_cast(dtype, device, config.supported_inference_dtypes)
    config.set_inference_dtype(dtype, manual, device=device)
    if model_options.get("fp8_optimizations", False): config.optimizations["fp8"] = True
    model = config.get_model(shape_sd, "", device=torch.device("meta"))
    root = model.diffusion_model
    model.device = device
    root.device = device
    lists, key_transform = _find_block_lists(root)
    logging.info("Lazy DiT: %s, block lists %s, key rewrite %s", type(root).__name__,
                 [name for name, _ in lists],
                 "flux _norm.scale" if key_transform is _checkpoint_key else "identity")
    n = sum(len(v) for _, v in lists)
    budget_bytes = int(memory_limit_mib * 2**20)
    k, nonblock_bytes, slot_bytes = _choose_k_for_budget(root, lists, budget_bytes)
    prefixes = tuple(name + "." for name, _ in lists)
    _load_nonblocks(root, reader, prefixes, key_transform=key_transform, scales=scales)
    c = math.ceil(k / 2)
    _, c, slot_positions = _slot_plan(lists, k)
    slot_store = _SlotStore()
    root.add_module("_lazy_dit_slots", slot_store)
    groups = {}
    for name, modules in lists:
        groups[name] = (modules, _BlockSlots(modules, device, slot_positions[name], slot_store), "", name)
    # patch_prefix: ModelPatcher keys are relative to the BaseModel, and
    # BaseModel holds the transformer as `diffusion_model`.
    pager = LazyBlockPager(root, reader, groups, k, debug, prefetch=prefetch,
                           key_transform=key_transform,
                           owner=model, patch_prefix="diffusion_model.")
    pager.scales = scales   # reuse the lookups _load_nonblocks already paid for
    root._lazy_dit_pager = pager   # lifetime + inspection handle
    model._lazy_pager = pager      # where LazyDiTModelPatcher.pager looks
    resident = sum(t.numel() * t.element_size() for t in root.parameters() if t.device.type != "meta")
    resident += sum(t.numel() * t.element_size() for t in root.buffers() if t.device.type != "meta")
    patcher = LazyDiTModelPatcher(model, device, device, resident)
    patcher.cached_patcher_init = (load_lazy_diffusion_model, (unet_path, model_options, memory_limit_mib, debug, prefetch))
    logging.info("Lazy DiT: limit=%.1f MiB, blocks=%s, k=%s, c=%s, planned=%.1f MiB, actual=%.1f MiB%s",
                 memory_limit_mib, n, k, c, (nonblock_bytes + slot_bytes) / 2**20, resident / 2**20,
                 f", dequantizing {quantized_layers} fp8 layers during paging" if quantized_layers else "")
    return patcher
