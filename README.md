# ComfyUI-LazyLoading-DiT

Lets you run FP16 Krea / Flux on Apple Silicon by streaming layers from SSD. No impact on quality because you're loading the full model.

You end up with Flux 2 Dev at FP8 consuming ~14GB including TE, Qwen-Image-2511 with a similar footprint, and Klein 9B FP16 taking up ~7GB including TE.

On my M5 Pro 16c this turned out to not be nearly as slow as it sounds: you spend a lot of time on compute anyway, which makes SSD streaming less painful especially if performed in the background. During test runs I observed up to about a ~10% slowdown per model, but YMMV, I haven't conducted proper benchmarks yet. Either way, this opens up the possibility to use large models in ComfyUI alongside LLMs or fit models that wouldn't otherwise fit.

---

Run **large diffusion transformer (DiT) models on a memory-tight Mac.**

This is a pair of drop-in ComfyUI nodes that load FLUX-family / Qwen-Image
models and their LLM text encoders **without ever holding the whole weights
file in RAM**. Instead of loading everything up front, it keeps only a small
set of transformer blocks resident at a time and **pages the rest in from SSD
during the forward pass** — so you can run checkpoints that wouldn't otherwise
fit in your Mac's unified memory.

Built for **Apple Silicon (MPS)**.

## Features

- **Run models that don't fit in RAM.** FLUX, Flux2 dev/Klein, FLUX.1 Krea,
  and Qwen-Image checkpoints can run even when the full weights exceed free
  unified memory. The full model is never materialized — not even transiently.
- **Cap memory with one number.** Set `memory_limit_mib` and the loader keeps
  exactly as many transformer blocks resident as that budget allows, paging
  the rest from disk on demand. Lower budget = less RAM, more SSD streaming.
- **MPS-native.** Designed around Apple Silicon's unified memory, where the
  real bottleneck is resident bytes, not host→device bandwidth. No CUDA
  required.
- **Lazy text encoders too.** Loads Qwen3 8B, Qwen3-VL 4B, Mistral3 24B, and
  Qwen2.5-VL 7B text encoders under the same memory budget.
- **Drop-in replacement.** Same `MODEL` / `CLIP` outputs and node category as
  the stock loaders — swap the node in, keep your workflow unchanged.
- **Handles quantized checkpoints.** Loads **fp8** and **int8**
  (`int8_tensorwise`) checkpoint files, dequantizing them on the fly during
  paging. File reads stay narrow (fp8/int8 width) so they use minimal disk I/O.
- **LoRA / patch friendly.** Model patches (e.g. LoRA) are re-applied to every
  streamed chunk automatically.
- **Fast paging.** Double-buffered prefetch on a background thread overlaps
  SSD reads with GPU compute, and direct uncached reads keep the kernel from
  pinning stale pages in RAM.

## Install

Copy this folder into `ComfyUI/custom_nodes/` and restart ComfyUI.

```
cp -r ComfyUI-LazyLoading-DiT <your-comfyui>/custom_nodes/
```

Then put your models in the usual folders:

- Diffusion checkpoints (`.safetensors`) → `ComfyUI/models/diffusion_models/`
- Text encoders (`.safetensors`) → `ComfyUI/models/text_encoders/`

Requires the `safetensors` package (already a ComfyUI dependency) and a PyTorch
build with MPS support.

## Quick start

1. Install the nodes (above) and drop a `.safetensors` model into
   `models/diffusion_models/`.
2. In ComfyUI, replace **Load Diffusion Model** with
   **Load Diffusion Model (Lazy DiT)**.
3. Set `memory_limit_mib` to how much RAM you can spare, leave `weight_dtype`
   at `default` (or pick `fp16`/`bf16`), and run — same as before.

## Nodes

### Load Diffusion Model (Lazy DiT)

Drop-in for the stock **Load Diffusion Model**. Outputs `MODEL`.

| Widget | Default | What it does |
|--------|---------|--------------|
| `unet_name` | — | Diffusion checkpoint (`.safetensors`) from `models/diffusion_models/`. |
| `weight_dtype` | `default` | `default`, `fp16`, `bf16`, `fp8_e4m3fn`, `fp8_e4m3fn_fast`, `fp8_e5m2`. **On MPS use `default`, `fp16` or `bf16`** (fp8 has no Metal matmul — see [Memory & dtypes](#memory--dtype-guidance)). |
| `memory_limit_mib` | 16384 | Max resident memory budget. Lower = fewer resident blocks, more SSD paging. |
| `debug` | off | Log per-chunk timing, resident-block count, and paging stats. |
| `prefetch` | on | Background-prefetch the next chunk while the current one computes. |

### Load Qwen Text Encoder (Lazy)

Drop-in for loading Qwen / Mistral text encoders. Outputs `CLIP`.

| Widget | Default | What it does |
|--------|---------|--------------|
| `clip_name` | — | Text-encoder file (`.safetensors`) from `models/text_encoders/`. |
| `memory_limit_mib` | 6144 | Max resident memory budget for the encoder. |
| `debug` | off | Log profile selection and paging details. |
| `profile` | `auto` | Encoder profile (below). `auto` detects, or refuses if ambiguous. |
| `prefetch` | on | Background-prefetch the next layer chunk. |

> `LazyFluxQwen3_8BLoader` is kept as an alias of the text-encoder loader so
> workflows saved against the earlier 8B-only release keep loading.

## What you can load

**Diffusion models** — block-paged by ComfyUI model class:

| Model class | Checkpoints | Blocks paged |
|-------------|-------------|--------------|
| `Flux` | FLUX family, Flux2 dev, Flux2 Klein | `double_blocks`, `single_blocks` |
| `SingleStreamDiT` | FLUX.1 Krea [dev] | `blocks` |
| `QwenImageTransformer2DModel` | Qwen-Image, Qwen-Image-Edit | `transformer_blocks` |

**Text encoders** — matched as `(checkpoint, consumer)` profiles, since one
file (e.g. Qwen3-VL 4B) can drive several encoders that each produce different
conditioning from identical weights. `auto` picks the profile that uniquely
matches the file, and refuses when more than one matches.

| Profile | Layers | Consumer / checkpoint |
|---------|--------|----------------------|
| `qwen3_8b_klein` | 36 | Qwen3 8B — Flux2 Klein |
| `qwen3vl_4b_krea2` | 36 | Qwen3-VL 4B — Krea 2 |
| `mistral3_24b_flux2` | 40 | Mistral3 24B — Flux2 dev |
| `mistral3_24b_pruned_flux2` | 30 | Mistral3 24B (pruned) — Flux2 dev |
| `qwen25_vl_7b_qwen_image` | 28 | Qwen2.5-VL 7B — Qwen-Image |

Mistral3 files also carry their tokenizer inside the weights (the `tekken_model`
key), which the profile accounts for.

## Memory & dtype guidance

**Recommended settings on Apple Silicon:** `weight_dtype = default`, `fp16`,
or `bf16`, and a `memory_limit_mib` you can comfortably spare (start with
`16384` for a large DiT and lower it until the machine stops swapping).

**fp8 / int8 support — what "supported" means here:**

- **Quantized *checkpoint files* are supported.** fp8 (`float8_e4m3fn` /
  `float8_e5m2`) and `int8_tensorwise` (non-rotated) checkpoints load fine.
  Their weights are dequantized on the fly during paging, so file reads stay
  fp8/int8-width (minimal disk I/O) while the resident slots hold the compute
  dtype. Only **per-tensor (scalar)** quantization is streamed.
- **fp8 *compute* is not available on MPS.** Metal has no fp8 matmul, so
  choosing an `fp8_*` `weight_dtype` on a Mac raises an error rather than
  crashing mid-render. Pick `fp16`/`bf16`/`default` instead — an fp8
  *checkpoint* still loads under those, because it is unpacked during paging.
- **int8 is only a storage format here**, not a `weight_dtype` option. It is
  supported when the checkpoint is `int8_tensorwise` without convolution
  rotation.
- Formats that need block scales or rotation — `nvfp4`, `mxfp8`,
  `convrot_w4a4`, `asym_w4a8_int8` — are **not** streamable and are rejected up
  front with a clear error.

**Resident memory** ≈ non-block parameters (embedders, norms, final layer,
modulation) + two chunks of block slots. Only a fraction of the model is in
memory at any instant.

## How it works

1. **Header-only detection.** Reads only the safetensors header (names/shapes/
   dtypes, a few KB) and builds the model on the `meta` device — **zero storage
   allocated**.
2. **Discovers block lists** in execution order (e.g. `double_blocks` then
   `single_blocks` for Flux); block parameter names are found at runtime, not
   hardcoded.
3. **Picks `k` for your budget.** Computes the largest non-full `k` whose
   typed slot allocation fits `memory_limit_mib`, always keeping at least two
   blocks absent.
4. **Pre-allocates slots.** `c = ceil(k/2)` slots per section, two sections (a
   double buffer) per block kind; every block parameter is redirected to its
   slot.
5. **Loads non-block params** once (embedders, norms, final layer,
   modulation) and keeps them resident.
6. **Pages during forward.** Chunk `j` lives in section `j % 2`; chunk 0 loads
   synchronously, chunk 1 prefetches in the background while chunk 0 computes,
   and so on — reset each full forward.
7. **Direct uncached SSD reads** (`pread`, `F_NOCACHE`) so the kernel doesn't
   pin streamed pages in RAM (an earlier mmap-based approach left ~23 GB of a
   26 GB checkpoint resident in mapped pages).
8. **All device writes on the forward thread**, in submission order — avoiding
   the cross-thread Metal command-buffer race that crashed earlier designs.

## Notes & limitations

- Requires a single `.safetensors` file for the diffusion model; the
  full-state-dict loader is never used.
- Slot aliasing is by design: blocks with congruent indices share storage at
  different points in one forward. A fresh tensor object is handed out per
  bind, so identity-keyed caches are safe.
- LoRA / model patches are re-applied to each chunk automatically.

## Repository layout

| File | Purpose |
|------|---------|
| `node.py` | `LazyLoadingDiTLoader` (Load Diffusion Model (Lazy DiT)). |
| `text_node.py` | `LazyQwenTextEncoderLoader` (Load Qwen Text Encoder (Lazy)). |
| `streaming.py` | Core engine: readers, `LazyBlockPager`, budget selection, architecture table. |
| `text_streaming.py` | Text-encoder paging: `PROFILES`, encoder loader, MPS fallbacks. |
| `test_lazy_loading_dit.py` | Manual MPS test plan: lazy vs stock loader across memory budgets. |
| `convert_dtype.py` | Standalone checkpoint dtype conversion utility. |

## License

See the repository for license terms.
