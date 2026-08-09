"""Lazy loader for single-file LLM text encoders (Flux.2 dev / Klein, Krea 2).

PROFILE TABLE -- keep checkpoint/layout facts here.  Every field below was read
out of a ComfyUI checkout (comfy/sd.py `load_text_encoder_state_dicts`,
comfy/text_encoders/{flux,krea2,qwen3vl,llama}.py, comfy/sd1_clip.py) at commit
2340099d9330, not inferred from upstream repos.

A profile is a (checkpoint layout, *consumer*) pair, not just a checkpoint.
Stock ComfyUI picks the target from `clip_type` as well as the detected
`TEModel`: one qwen3vl_4b file drives Krea 2 (12-layer tap, `krea2.te`), Flux.2
Klein (3-layer tap, `klein_te(model_type="qwen3_4b")`), Mage-Flow and the
generic `qwen3vl.te`, and those produce different conditioning from identical
weights.  So the checkpoint alone cannot select a profile; `auto` refuses when
more than one registered profile matches the file.

| profile                   | ckpt keys ->                | layers | siblings | tokenizer_data |
| qwen3_8b_klein            | model.*                     | 36     | --       | --             |
| qwen3vl_4b_krea2          | model.language_model.* etc. | 36     | visual   | --             |
| mistral3_24b_flux2        | model.*                     | 40     | --       | tekken_model   |
| mistral3_24b_pruned_flux2 | model.*                     | 30     | --       | tekken_model   |
| qwen25_vl_7b_qwen_image   | model.*                     | 28     | visual   | --             |

The Mistral entries are the Flux.2 *dev* encoder, not Klein's: sd.py routes
TEModel.MISTRAL3_24B to flux2_te + Flux2Tokenizer with no clip_type fork, so
unlike the Qwen files the checkpoint alone does settle the consumer.  Its
tokenizer is serialized into the weights file (`tekken_model`) instead of
shipping with ComfyUI, which is why tokenizer_data is a profile field.

The wrapper attribute is not hardcoded: `sd1_clip.SD1ClipModel` stores its own
child's name in `self.clip` ("qwen3_8b" when constructed with `name=`,
"clip_l"-style otherwise), and this module reads it the same way ComfyUI does.
"""
from __future__ import annotations

import dataclasses
import logging
import re
import types
from typing import Callable

import torch

import comfy.model_management
import comfy.sd
import comfy.text_encoders.flux
import comfy.text_encoders.krea2
import comfy.text_encoders.qwen_image
import comfy.hooks
import folder_paths

from .streaming import (
    _QUANTIZED_STORAGE,
    _audit_quantization,
    _quantization_scale,
    LazyBlockPager,
    LazyDiTModelPatcher,
    _BlockSlots,
    _SlotStore,
    _TensorReader,
    _choose_k_for_budget,
    _dtype_copy_,
    _identity_key,
    _load_nonblocks,
    _named_local_tensors,
    _set_qualified,
    _slot_plan,
)


@dataclasses.dataclass(frozen=True)
class _QwenTEProfile:
    """One audited (checkpoint layout, ComfyUI consumer) pair."""

    name: str
    consumer: str                                   # which DiT the conditioning is for
    build_target: Callable[[torch.dtype], types.SimpleNamespace]
    text_module_path: str                           # path under <wrapper>.transformer
    block_list_name: str
    checkpoint_prefix: str                          # prepended to text-module keys
    expected_layers: int
    embedding_key: str
    embedding_shape: tuple
    # Accepted storage dtypes, not a pinned one: the modules are constructed
    # with whatever dtype the checkpoint carries (see dtype_keys below), so
    # fp16 and bf16 releases of the same encoder are the same profile.  This
    # is a sanity filter, not a discriminator -- embedding shape and key
    # layout already separate every registered profile.
    embedding_dtypes: tuple
    # llama_detect's dtype probe, in its order: model.norm.weight first, then
    # model.layers.0.input_layernorm.weight.  A tuple rather than one key
    # because a pruned encoder is built with final_norm=False and its file
    # carries no model.norm.weight at all.
    dtype_keys: tuple
    probe_keys: tuple
    # Keys whose *absence* is part of the identity.  The full and pruned
    # Mistral releases are byte-identical in every probed key the pruned one
    # has; detect_te_model separates them on model.layers.39 alone.
    absent_keys: tuple = ()
    probe_prefixes: tuple = ()
    # Tensors the ComfyUI module builds that the checkpoint legitimately omits.
    # Stock load_sd is strict=False, so these only downgrade a warning to a
    # debug line; anything else missing is still reported loudly.
    optional_keys: tuple = ()                      # at least one key under each
    # comfy.utils.state_dict_prefix_replace, as applied by sd.py before load.
    prefix_replace: tuple = ()                      # ((checkpoint prefix, model prefix), ...)
    # Sibling modules of the text model that must be materialized but are not
    # paged: (module path under transformer, checkpoint prefix).
    resident_siblings: tuple = ()
    # The Flux DiT rewrite of *_norm.weight -> *_norm.scale is specific to
    # comfy.supported_models.Flux.process_unet_state_dict.  No Qwen text
    # encoder uses it: llama.py's RMSNorm parameter is `weight`, and Qwen3
    # additionally has self_attn.{q,k}_norm.weight, which that rewrite would
    # mistranslate into keys no checkpoint has.
    key_transform: Callable[[str], str] = _identity_key
    # sd.py sets this for every text encoder (CLIP.__init__ ->
    # set_model_compute_dtype(torch.float32)); it is not a per-profile choice
    # so far, but it is a per-profile fact, so it lives with the rest of them.
    compute_dtype: torch.dtype = torch.float32
    # Checkpoint keys that sd.py copies into tokenizer_data rather than
    # loading as weights -- a serialized tokenizer carried inside the
    # safetensors.  Without these the tokenizer cannot be constructed at all.
    tokenizer_data_keys: tuple = ()
    aliases: tuple = ()


def _klein_8b_target(dtype):
    """sd.py: TEModel.QWEN3_8B, clip_type FLUX/FLUX2 (not IDEOGRAM4)."""
    return types.SimpleNamespace(
        params={},
        clip=comfy.text_encoders.flux.klein_te(dtype_llama=dtype, model_type="qwen3_8b"),
        tokenizer=comfy.text_encoders.flux.KleinTokenizer8B,
    )


def _krea2_target(dtype):
    """sd.py: TEModel.QWEN3VL_4B with clip_type == CLIPType.KREA2.

    This is the branch that matters for conditioning: Krea2Qwen3VLClipModel
    taps hidden states [2,5,...,35] and Krea2TEModel flattens them to
    (B, seq, 12*2560) for the DiT's txtfusion adapter.  The Flux2 Klein branch
    for the same file taps 3 layers instead, so picking the wrong one produces
    a wrong-width, wrong-content conditioning tensor rather than an error.
    """
    return types.SimpleNamespace(
        params={},
        clip=comfy.text_encoders.krea2.te(dtype_llama=dtype),
        tokenizer=comfy.text_encoders.krea2.Krea2Tokenizer,
    )


QWEN3_8B_KLEIN = _QwenTEProfile(
    name="qwen3_8b_klein",
    consumer="Flux.2 Klein",
    build_target=_klein_8b_target,
    text_module_path="model",
    block_list_name="layers",
    checkpoint_prefix="model.",
    expected_layers=36,
    embedding_key="model.embed_tokens.weight",
    embedding_shape=(151936, 4096),
    embedding_dtypes=(torch.float16, torch.bfloat16),
    dtype_keys=("model.norm.weight", "model.layers.0.input_layernorm.weight"),
    probe_keys=("model.embed_tokens.weight", "model.norm.weight", "model.layers.35.self_attn.q_proj.weight"),
    # sd.py applies this to every text encoder before detection ("prefix
    # missing in some models"); harmless when the file already has neither.
    prefix_replace=(("lm_head.", "model.lm_head."),),
    # Qwen3_8BConfig sets lm_head=True so Llama2_ builds one, but the released
    # Klein encoders tie word embeddings and ship no lm_head.  Only
    # Llama2_.generate reads it; conditioning never does.
    optional_keys=("model.lm_head.weight",),
    aliases=("qwen3_8b_fp16", "qwen3_8b_fp16_klein"),
)

QWEN3VL_4B_KREA2 = _QwenTEProfile(
    name="qwen3vl_4b_krea2",
    consumer="Krea 2",
    build_target=_krea2_target,
    text_module_path="model",
    block_list_name="layers",
    checkpoint_prefix="model.",
    expected_layers=36,          # llama.Qwen3VL_4BConfig <- Qwen3_8BConfig.num_hidden_layers
    embedding_key="model.embed_tokens.weight",
    embedding_shape=(151936, 2560),   # Qwen3VL_4BConfig.hidden_size = 2560
    embedding_dtypes=(torch.bfloat16, torch.float16),
    dtype_keys=("model.norm.weight", "model.layers.0.input_layernorm.weight"),
    probe_keys=("model.embed_tokens.weight", "model.norm.weight", "model.layers.35.self_attn.q_norm.weight"),
    probe_prefixes=("visual.",),
    # sd.py applies exactly this before load_sd for every QWEN3VL_4B branch.
    prefix_replace=(("model.language_model.", "model."), ("model.visual.", "visual."), ("lm_head.", "model.lm_head.")),
    # Qwen3VL.__init__ always builds self.visual, and Krea2 does accept images,
    # so the tower must be materialized even though text-only prompts never run
    # it.  It is resident for now; paging it needs its own schedule because
    # Qwen35VisionModel.forward is a separate, conditionally-executed root.
    resident_siblings=(("visual", "visual."),),
    aliases=("qwen3vl_4b_bf16_krea2",),
)

def _flux2_mistral_target(dtype, pruned):
    """sd.py: TEModel.MISTRAL3_24B / MISTRAL3_24B_PRUNED_FLUX2.

    Unlike the Qwen entries there is no clip_type fork here -- both TEModel
    values route to flux2_te + Flux2Tokenizer unconditionally -- so the
    checkpoint really does determine the consumer, and `auto` can settle it.

    Mistral3_24BModel taps layer=[10, 20, 30] and Flux2TEModel stacks the three
    into (B, seq, 3*5120) for Flux.2 dev's txt_in.  The tap is not an early
    exit: Llama2_.forward runs all 40 layers regardless, so every one of them
    still has to be paged.
    """
    return types.SimpleNamespace(
        params={},
        clip=comfy.text_encoders.flux.flux2_te(dtype_llama=dtype, pruned=pruned),
        tokenizer=comfy.text_encoders.flux.Flux2Tokenizer,
    )


MISTRAL3_24B_FLUX2 = _QwenTEProfile(
    name="mistral3_24b_flux2",
    consumer="Flux.2 dev",
    build_target=lambda dtype: _flux2_mistral_target(dtype, pruned=False),
    text_module_path="model",
    block_list_name="layers",
    checkpoint_prefix="model.",
    expected_layers=40,          # llama.Mistral3Small24BConfig.num_hidden_layers
    embedding_key="model.embed_tokens.weight",
    embedding_shape=(131072, 5120),
    embedding_dtypes=(torch.bfloat16, torch.float16),
    dtype_keys=("model.norm.weight", "model.layers.0.input_layernorm.weight"),
    # detect_te_model reaches MISTRAL3_24B by post_attention_layernorm width
    # 5120 *without* a self_attn.q_norm (Mistral3Small24BConfig sets q_norm =
    # k_norm = None); layer 39 is what separates full from pruned.
    probe_keys=(
        "model.embed_tokens.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.39.post_attention_layernorm.weight",
        "model.norm.weight",
    ),
    absent_keys=("model.layers.0.self_attn.q_norm.weight",),
    # Mistral3Small24BConfig.lm_head is False, so Llama2_ builds none and the
    # checkpoint is expected to carry none; nothing is optional here.
    # sd.py's MISTRAL3 branch applies no state_dict_prefix_replace.
    tokenizer_data_keys=("tekken_model",),
    aliases=("mistral_3_small_flux2", "mistral_3_small_flux2_bf16", "mistral3_24b"),
)

MISTRAL3_24B_PRUNED_FLUX2 = dataclasses.replace(
    MISTRAL3_24B_FLUX2,
    name="mistral3_24b_pruned_flux2",
    consumer="Flux.2 dev (pruned TE)",
    build_target=lambda dtype: _flux2_mistral_target(dtype, pruned=True),
    # flux2_te(pruned=True) -> model_options["num_layers"] = 30, and
    # Mistral3_24BModel sets final_norm=False below 40, so this graph has no
    # model.norm at all -- hence the dtype probe order and the absent key.
    expected_layers=30,
    probe_keys=(
        "model.embed_tokens.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.29.post_attention_layernorm.weight",
    ),
    absent_keys=(
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.39.post_attention_layernorm.weight",
    ),
    aliases=("mistral_3_small_flux2_pruned",),
)

def _qwen_image_target(dtype):
    """sd.py: TEModel.QWEN25_7B with clip_type neither HUNYUAN_IMAGE nor LONGCAT_IMAGE.

    Same encoder and same tokenizer for Qwen-Image and every Qwen-Image-Edit
    revision; QwenImageTokenizer picks its template at tokenize time from
    whether images were passed, not from the checkpoint, so 2509 and 2511 are
    one profile here.  The two clip_type forks that would change the consumer
    (Hunyuan Image, LongCat) are different tokenizers entirely, and neither is
    registered, so `auto` can settle this file.
    """
    return types.SimpleNamespace(
        params={},
        clip=comfy.text_encoders.qwen_image.te(dtype_llama=dtype),
        tokenizer=comfy.text_encoders.qwen_image.QwenImageTokenizer,
    )


QWEN25_VL_7B_QWEN_IMAGE = _QwenTEProfile(
    name="qwen25_vl_7b_qwen_image",
    consumer="Qwen-Image / Qwen-Image-Edit",
    build_target=_qwen_image_target,
    text_module_path="model",
    block_list_name="layers",
    checkpoint_prefix="model.",
    expected_layers=28,               # llama.Qwen25_7BVLI_Config.num_hidden_layers
    embedding_key="model.embed_tokens.weight",
    embedding_shape=(152064, 3584),
    embedding_dtypes=(torch.bfloat16, torch.float16),
    dtype_keys=("model.norm.weight", "model.layers.0.input_layernorm.weight"),
    # Qwen25_7BVLI_Config sets qkv_bias=True, which no other registered profile
    # does -- Qwen3 and Mistral both build their projections without bias -- so
    # the q_proj bias is the cheapest positive discriminator available.
    probe_keys=(
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.27.self_attn.q_proj.weight",
    ),
    # ...and q_norm is None here, unlike Qwen3.
    absent_keys=("model.layers.0.self_attn.q_norm.weight",),
    probe_prefixes=("visual.",),
    # Applied to every text encoder by sd.py before detection; the QWEN25_7B
    # branch adds none of its own.
    prefix_replace=(("lm_head.", "model.lm_head."),),
    # Qwen25_7BVLI always builds self.visual, and unlike Krea 2 the Edit
    # workflows actually run it -- the image goes through the vision tower and
    # its output is spliced in at the <|image_pad|> token.  Resident, not
    # paged: Qwen2VLVisionTransformer.forward is a separate root that runs at
    # most once per prompt, so paging it would buy little and needs its own
    # schedule.
    resident_siblings=(("visual", "visual."),),
    aliases=("qwen25_vl_7b", "qwen_2.5_vl_7b", "qwen_image", "qwen_image_edit",
             "qwen-image-2511-edit", "qwen_image_edit_2511"),
)

PROFILES = {p.name: p for p in (QWEN3_8B_KLEIN, QWEN3VL_4B_KREA2,
                                MISTRAL3_24B_FLUX2, MISTRAL3_24B_PRUNED_FLUX2,
                                QWEN25_VL_7B_QWEN_IMAGE)}
_ALIASES = {alias: p.name for p in PROFILES.values() for alias in p.aliases}


def _install_mps_area_interpolate_fallback():
    """Make Qwen 2.5-VL's non-divisible ``area`` resize work on MPS.

    ``F.interpolate(..., mode="area")`` is implemented by adaptive average
    pooling.  MPS currently only implements that pool when each input spatial
    size is divisible by its requested output size.  Qwen's image grid can
    legitimately produce other ratios, so retry exactly that unsupported case
    on CPU and put the small resized result back on the original device.

    This is process-wide because the failing resize happens in ComfyUI's Qwen
    vision code, not in this node.  The wrapper is otherwise transparent and
    deliberately does not turn arbitrary MPS errors into CPU work.
    """
    functional = torch.nn.functional
    current = functional.interpolate
    if getattr(current, "_lazy_qwen_mps_area_fallback", False):
        return

    def interpolate(input, *args, **kwargs):
        try:
            return current(input, *args, **kwargs)
        except RuntimeError as error:
            if (input.device.type != "mps" or
                    "Adaptive pool MPS: input sizes must be divisible by output sizes" not in str(error)):
                raise
            logging.debug(
                "Lazy Qwen TE: retrying non-divisible MPS area interpolation on CPU "
                "(input=%s, size=%s)",
                tuple(input.shape), kwargs.get("size", args[0] if args else None),
            )
            return current(input.to("cpu"), *args, **kwargs).to(input.device)

    interpolate._lazy_qwen_mps_area_fallback = True
    functional.interpolate = interpolate


# --------------------------------------------------------------------------
# Checkpoint views and inspection (header only -- no large tensor is read).
# --------------------------------------------------------------------------

class _RenamedReader:
    """Reader whose keys are the model-side keys sd.py loads from.

    comfy.utils.state_dict_prefix_replace rewrites the state dict in place and
    in dict order; this applies the same pairs longest-prefix-first, which
    agrees with it for every registered profile and is order-independent.
    """

    def __init__(self, reader, prefix_replace):
        self._reader = reader
        self.path = getattr(reader, "path", None)
        pairs = sorted(prefix_replace, key=lambda pair: len(pair[0]), reverse=True)
        mapping = {}
        for key in reader.keys():
            for source, target in pairs:
                if key.startswith(source):
                    mapping[target + key[len(source):]] = key
                    break
            else:
                mapping[key] = key
        self.mapping = mapping

    def keys(self):
        return self.mapping.keys()

    def get_tensor(self, key):
        return self._reader.get_tensor(self.mapping[key])

    def get_slice(self, key):
        return self._reader.get_slice(self.mapping[key])

    def byte_range(self, key):
        return self._reader.byte_range(self.mapping[key])

    def metadata(self):
        return self._reader.metadata()


def _view(reader, profile):
    return _RenamedReader(reader, profile.prefix_replace) if profile.prefix_replace else reader


_LAYER_RE = re.compile(r"^(?P<stem>.*?)(?P<list>[A-Za-z_][A-Za-z0-9_]*)\.(?P<index>\d+)\.")


def _shape_and_dtype(reader, key):
    """Shape/dtype of one checkpoint tensor without reading its data."""
    sl = reader.get_slice(key)
    shape = tuple(sl.get_shape())
    if not shape or not shape[0]:
        return shape, None
    # One row is enough to learn the dtype; loading the whole embedding here
    # would cost gigabytes on exactly the machines this node exists for.
    return shape, sl[0:1].dtype


def describe_checkpoint(reader):
    """Human-readable summary used by detection errors and by audit_qwen_te."""
    keys = list(reader.keys())
    lines = []
    for key in [k for k in keys if k.endswith("embed_tokens.weight")]:
        shape, dtype = _shape_and_dtype(reader, key)
        lines.append(f"  {key}: shape={shape} dtype={dtype}")
    lists = {}
    for key in keys:
        match = _LAYER_RE.match(key)
        if match:
            lists.setdefault(match.group("stem") + match.group("list"), set()).add(int(match.group("index")))
    for stem in sorted(lists):
        lines.append(f"  block list {stem!r}: {len(lists[stem])} entries")
    lines.append(f"  top-level prefixes: {sorted({key.split('.')[0] for key in keys})}")
    return "\n".join(lines)


def _profile_mismatch(profile, reader):
    """None if this profile matches the checkpoint, else why it does not."""
    view = _view(reader, profile)
    missing = [key for key in profile.probe_keys if key not in view.keys()]
    if missing:
        return "missing " + ", ".join(missing)
    present = [key for key in profile.absent_keys if key in view.keys()]
    if present:
        return "has " + ", ".join(present) + ", which this profile's checkpoint does not"
    for prefix in profile.probe_prefixes:
        if not any(key.startswith(prefix) for key in view.keys()):
            return f"no key under {prefix!r}"
    shape, dtype = _shape_and_dtype(view, profile.embedding_key)
    if shape != profile.embedding_shape:
        return f"embedding {profile.embedding_key} is {shape}, expected {profile.embedding_shape}"
    if dtype not in profile.embedding_dtypes and dtype not in _QUANTIZED_STORAGE:
        # A quantized storage dtype is accepted for every profile, not just the
        # ones that list it.
        # Storage dtype was never a discriminator here -- key layout and
        # embedding shape already separate every registered profile -- and a
        # quantized release of any of these encoders has fp8 in this slot.
        # The dequantized module dtype comes from dtype_keys, not from here.
        return (f"embedding {profile.embedding_key} is {dtype}, "
                f"expected one of {profile.embedding_dtypes}")
    return None


def _select_profile(reader, requested):
    if requested and requested != "auto":
        name = _ALIASES.get(requested, requested)
        profile = PROFILES.get(name)
        if profile is None:
            raise RuntimeError(f"Unknown Qwen TE profile {requested!r}; known: {sorted(PROFILES)}.")
        reason = _profile_mismatch(profile, reader)
        if reason:
            raise RuntimeError(f"Checkpoint does not match profile {name!r}: {reason}.")
        return profile
    matches, reasons = [], {}
    for name, profile in PROFILES.items():
        reason = _profile_mismatch(profile, reader)
        if reason is None:
            matches.append(profile)
        else:
            reasons[name] = reason
    if len(matches) == 1:
        return matches[0]
    if matches:
        # Same weights, different tap/template -> different conditioning.  This
        # is not a detail auto-detection is allowed to guess.
        options = ", ".join(f"{p.name} ({p.consumer})" for p in matches)
        raise RuntimeError(
            "This checkpoint matches more than one Qwen TE profile, and they produce different "
            f"conditioning from the same weights. Set `profile` explicitly: {options}."
        )
    detail = "\n".join(f"  {name}: {reason}" for name, reason in reasons.items())
    raise RuntimeError(
        "Lazy Qwen TE has no audited profile for this checkpoint.\n"
        f"Checked profiles:\n{detail}\n"
        f"Checkpoint looks like:\n{describe_checkpoint(reader)}\n"
        "Add a _QwenTEProfile to text_streaming.py; audit_qwen_te.py prints its fields."
    )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _move_outer_non_checkpoint_parameters(outer, device, skip_prefix):
    """SDClipModel.logit_scale is initialized, not present in qwen safetensors."""
    for name, tensor in list(outer.named_parameters(recurse=True)):
        # The wrapper owns ``transformer``, whose tensors are resident text
        # non-blocks, resident siblings, or paged layers; never copied here.
        if name.startswith(skip_prefix):
            continue
        if tensor.device.type == "meta":
            raise RuntimeError(f"Unexpected meta tensor outside the Qwen model: {name}")
        # Loader nodes can run inside inference_mode(), but this resident
        # parameter later participates in ordinary ComfyUI execution.
        with torch.inference_mode(False):
            moved = torch.nn.Parameter(tensor.detach().to(device), requires_grad=False)
        _set_qualified(outer, name, moved)


def _load_sibling(module, reader, checkpoint_prefix, key_transform, device, scales=None):
    """Materialize a non-paged sibling module (the Qwen3-VL vision tower).

    Deliberately tolerant, unlike streaming._load_nonblocks: a tensor the
    checkpoint does not carry is left on meta and reported, so a text-only
    workflow still loads.  Meta tensors raise on use, so an incomplete tower
    fails loudly at the first image rather than producing wrong pixels.
    """
    parameters = dict(module.named_parameters(recurse=True))
    buffers = dict(module.named_buffers(recurse=True))
    if scales is None:
        scales = {}
    loaded_bytes, skipped = 0, []
    for name, tensor in list(parameters.items()) + list(buffers.items()):
        key = checkpoint_prefix + key_transform(name)
        if key not in reader.keys():
            # Non-persistent buffers are computed in __init__ and are in no
            # checkpoint by design.  Qwen35VisionRotaryEmbedding.inv_freq is
            # built with torch.arange(..., dtype=torch.float) and no device
            # argument, so it lands on CPU, not meta, and its forward moves the
            # result to the input's device.  Only a meta straggler is a problem.
            skipped.append((name, tensor.device.type))
            continue
        with torch.inference_mode(False):
            destination = torch.empty(tuple(tensor.shape), dtype=tensor.dtype, device=device)
        source = reader.get_tensor(key)
        # The scale matters here as much as anywhere: a released encoder can
        # quantize its vision tower alongside the language model, and a
        # dequantization that skipped the scale would produce plausible-looking
        # numbers off by a constant factor rather than an error.
        _dtype_copy_(destination, source, _quantization_scale(reader, key, scales))
        del source
        _set_qualified(
            module, name,
            torch.nn.Parameter(destination, requires_grad=False) if name in parameters else destination,
            buffer=name in buffers,
        )
        loaded_bytes += destination.numel() * destination.element_size()
    return loaded_bytes, skipped


def _assert_streamed_keys_present(reader, layers, profile, block_prefix):
    """Fail at load time, not mid-forward, if any streamed key is absent."""
    missing = []
    for index in range(len(layers)):
        for local_name, _, _ in _named_local_tensors(layers[index]):
            key = profile.key_transform(f"{block_prefix}{index}.{local_name}")
            if key not in reader.keys():
                missing.append(key)
    if missing:
        raise RuntimeError(
            f"Lazy Qwen TE profile {profile.name!r} expects {len(missing)} streamed tensor(s) the "
            f"checkpoint does not have, e.g. {missing[:4]}. The profile's key_transform, "
            "checkpoint_prefix or prefix_replace is wrong for this file."
        )


def load_lazy_qwen_text_encoder(clip_path, memory_limit_mib, debug=False, profile_name="auto", prefetch=True):
    if not clip_path.lower().endswith(".safetensors"):
        raise RuntimeError("Lazy Qwen TE requires one .safetensors file.")
    raw = _TensorReader(clip_path)
    profile = _select_profile(raw, profile_name)
    if profile is QWEN25_VL_7B_QWEN_IMAGE:
        _install_mps_area_interpolate_fallback()
    reader = _view(raw, profile)
    device = comfy.model_management.get_torch_device()
    if device.type != "mps":
        raise RuntimeError(f"Lazy Qwen TE is MPS-only; current ComfyUI device is {device}.")

    # hunyuan_video.llama_detect takes dtype from the final norm, not the
    # embedding, and falls through to layer 0's input norm; mirror both, in
    # its order, so the constructed modules match the file.
    dtype_llama = None
    for key in profile.dtype_keys:
        if key in reader.keys():
            _, dtype_llama = _shape_and_dtype(reader, key)
            break
    if dtype_llama is None:
        raise RuntimeError(
            f"Lazy TE profile {profile.name!r}: none of its dtype probes {profile.dtype_keys} "
            "is in the checkpoint, so the module dtype cannot be matched to the file."
        )
    if dtype_llama in _QUANTIZED_STORAGE:
        # Only reachable if a release quantizes its RMSNorm weights, which none
        # of the registered ones do.  Building the modules at a storage dtype
        # would put fp8 or int8 tensors on Metal as if they were compute types;
        # the file is still perfectly loadable, it just has to land in a real
        # compute dtype.
        logging.warning("Lazy TE: %s is %s; building the modules bf16 and dequantizing on copy.",
                        profile.dtype_keys[0], dtype_llama)
        dtype_llama = torch.bfloat16

    # Reject quantization formats the copy-time dequantizer cannot express
    # before anything large is allocated.  Note that no quantization metadata
    # is passed to build_target: the modules are built dense, and the fp8
    # values are unpacked during the copy, exactly as on the DiT side.
    scales = {}
    quantized_layers = _audit_quantization(raw)

    # Some tokenizers are serialized *into* the weights file and sd.py hands
    # them to CLIP through tokenizer_data rather than loading them as tensors
    # (Mistral's tekken vocab is a uint8 blob; load_mistral_tokenizer accepts
    # the tensor directly).  These are small, so reading them eagerly is fine,
    # and without them the tokenizer cannot be constructed at all.
    tokenizer_data = {}
    for key in profile.tokenizer_data_keys:
        if key not in reader.keys():
            raise RuntimeError(
                f"Lazy TE profile {profile.name!r} needs {key!r} for its tokenizer, but this "
                "checkpoint has no such key. ComfyUI's own loader would fail the same way."
            )
        tokenizer_data[key] = reader.get_tensor(key)

    # Construct only the stock wrapper/tokenizer graph on meta.  Passing a state
    # dict to comfy.sd.load_clip would materialize every text-model tensor, so
    # do not.
    clip = comfy.sd.CLIP(
        target=profile.build_target(dtype_llama),
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        parameters=0,
        tokenizer_data=tokenizer_data,
        model_options={"load_device": device, "offload_device": device, "initial_device": torch.device("meta")},
    )
    outer = clip.cond_stage_model
    # sd1_clip.SD1ClipModel stores its child's attribute name in self.clip and
    # reaches it with getattr(self, self.clip); do the same rather than guessing
    # between the "qwen3_8b" and "clip_qwen3_8b" naming branches.
    wrapper_attr = getattr(outer, "clip", None)
    if not isinstance(wrapper_attr, str) or not hasattr(outer, wrapper_attr):
        children = [name for name, _ in outer.named_children()]
        raise RuntimeError(f"Unsupported ComfyUI layout: {type(outer).__name__}.clip is {wrapper_attr!r}; children {children}.")
    transformer = getattr(outer, wrapper_attr).transformer
    text_root = transformer.get_submodule(profile.text_module_path) if profile.text_module_path else transformer
    text_root.device = device

    layers = getattr(text_root, profile.block_list_name, None)
    if not isinstance(layers, torch.nn.ModuleList) or len(layers) != profile.expected_layers:
        found = [(name, len(item)) for name, item in text_root.named_children() if isinstance(item, torch.nn.ModuleList)]
        raise RuntimeError(
            f"Unsupported ComfyUI layout for profile {profile.name!r}: expected {profile.expected_layers} "
            f"{profile.text_module_path}.{profile.block_list_name}; found ModuleLists {found}."
        )

    skip_prefix = f"{wrapper_attr}.transformer."
    _move_outer_non_checkpoint_parameters(outer, device, skip_prefix)
    missing = _load_nonblocks(
        text_root, reader, (profile.block_list_name + ".",),
        checkpoint_prefix=profile.checkpoint_prefix, key_transform=profile.key_transform, strict=False,
        scales=scales,
    )
    expected = [key for key in missing if key in profile.optional_keys]
    unexpected = [key for key in missing if key not in profile.optional_keys]
    if expected:
        logging.debug("Lazy Qwen TE missing (expected for %s): %s", profile.name, expected)
    if unexpected:
        # Same policy as sd.py: warn and keep going.  These stay on meta, so
        # any code path that reads them raises rather than reading garbage.
        logging.warning("Lazy Qwen TE missing: %s", unexpected)

    sibling_bytes = 0
    for path, prefix in profile.resident_siblings:
        module = transformer.get_submodule(path)
        module.device = device
        loaded, skipped = _load_sibling(module, reader, prefix, profile.key_transform, device, scales=scales)
        sibling_bytes += loaded
        logging.info("Lazy Qwen TE: resident sibling %r = %.1f MiB", path, loaded / 2**20)
        stranded = [name for name, kind in skipped if kind == "meta"]
        derived = [name for name, kind in skipped if kind != "meta"]
        if derived:
            logging.info(
                "Lazy Qwen TE: sibling %r has %d derived tensor(s) not in the checkpoint, already "
                "materialized by __init__: %s", path, len(derived), derived[:8],
            )
        if stranded:
            logging.warning(
                "Lazy Qwen TE: %d tensor(s) in sibling %r are absent from the checkpoint and left on meta "
                "(text-only prompts are unaffected; any use raises): %s%s",
                len(stranded), path, stranded[:8], " ..." if len(stranded) > 8 else "",
            )

    block_prefix = f"{profile.checkpoint_prefix}{profile.block_list_name}."
    _assert_streamed_keys_present(reader, layers, profile, block_prefix)

    lists = [(profile.block_list_name, layers)]
    budget_bytes = int(memory_limit_mib * 2**20) - sibling_bytes
    if budget_bytes <= 0:
        raise RuntimeError(
            f"Lazy Qwen TE memory_limit_mib={memory_limit_mib:.1f} is smaller than the resident "
            f"sibling modules alone ({sibling_bytes / 2**20:.1f} MiB)."
        )
    k, permanent_bytes, slot_bytes = _choose_k_for_budget(text_root, lists, budget_bytes)
    _, c, slot_positions = _slot_plan(lists, k)
    slot_store = _SlotStore()
    text_root.add_module("_lazy_qwen_slots", slot_store)
    groups = {profile.block_list_name: (
        layers,
        _BlockSlots(layers, device, slot_positions[profile.block_list_name], slot_store),
        "",
        profile.block_list_name,
    )}
    pager = LazyBlockPager(
        text_root, reader, groups, k, debug,
        checkpoint_prefix=profile.checkpoint_prefix, key_transform=profile.key_transform, prefetch=prefetch,
        # `owner` stays None: nothing calls pre_run on a CLIP patcher, so there
        # is no current_patcher to read.  patch_prefix is still needed, because
        # a CLIP LoRA's keys are relative to the wrapper, and it is what tells
        # the patcher which patched keys are paged and which are resident.
        patch_prefix=f"{skip_prefix}{profile.text_module_path}." if profile.text_module_path else skip_prefix,
    )
    pager.scales = scales   # reuse the lookups _load_nonblocks already paid for
    text_root._lazy_qwen_pager = pager
    outer._lazy_pager = pager   # where LazyDiTModelPatcher.pager looks

    resident = sum(t.numel() * t.element_size() for t in outer.parameters() if t.device.type != "meta")
    resident += sum(t.numel() * t.element_size() for t in outer.buffers() if t.device.type != "meta")
    # Base ModelPatcher signature: (model, load_device, offload_device, size).
    # Not (model, device, size) -- clone() reconstructs through this signature.
    patcher = LazyDiTModelPatcher(outer, device, device, resident)
    patcher.set_model_compute_dtype(profile.compute_dtype)
    patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
    patcher.is_clip = True
    clip.patcher = patcher
    logging.info(
        "Lazy TE [%s -> %s]: limit=%.1f MiB, dtype=%s, layers=%s, k=%s, c=%s, "
        "siblings=%.1f MiB, planned=%.1f MiB, actual=%.1f MiB%s",
        profile.name, profile.consumer, memory_limit_mib, dtype_llama, len(layers), k, c,
        sibling_bytes / 2**20, (permanent_bytes + slot_bytes + sibling_bytes) / 2**20, resident / 2**20,
        f", dequantizing {quantized_layers} fp8 layers during paging" if quantized_layers else "",
    )
    return clip


def load_lazy_flux_qwen3_8b(clip_path, memory_limit_mib, debug=False):
    """Backwards-compatible entry point for the original 8B-only node."""
    return load_lazy_qwen_text_encoder(clip_path, memory_limit_mib, debug, profile_name="qwen3_8b_klein")
