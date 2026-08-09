import torch

import folder_paths
from .streaming import load_lazy_diffusion_model


class LazyLoadingDiTLoader:
    """API-compatible Load Diffusion Model, with an MPS-oriented block pager."""

    @classmethod
    def INPUT_TYPES(cls):
        # Matches ComfyUI nodes.py:UNETLoader.INPUT_TYPES, with two additions.
        return {"required": {
            "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
            # "default" asks comfy.model_management.unet_dtype to choose, which
            # can land on fp16 for a bf16 checkpoint.  Same width, no saving,
            # but it makes every slot copy an elementwise conversion instead of
            # a blit.  fp16/bf16 let you match the file.  Combo widgets store
            # their value as a string, so adding options is safe for saved
            # workflows as long as the old strings remain.
            "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "fp16", "bf16"], {"advanced": True}),
            "memory_limit_mib": ("FLOAT", {"default": 16384.0, "min": 256.0, "max": 262144.0, "step": 256.0}),
            "debug": ("BOOLEAN", {"default": False}),
            # Appended, not inserted: saved workflows store widget values
            # positionally.
            "prefetch": ("BOOLEAN", {"default": True, "advanced": True}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "model/loaders"

    def load_unet(self, unet_name, weight_dtype, memory_limit_mib=16384.0, debug=False, prefetch=True):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        elif weight_dtype == "fp16":
            model_options["dtype"] = torch.float16
        elif weight_dtype == "bf16":
            model_options["dtype"] = torch.bfloat16

        # Exact stock resolution: folder_paths.get_full_path_or_raise("diffusion_models", name).
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        return (load_lazy_diffusion_model(unet_path, model_options, float(memory_limit_mib), bool(debug), bool(prefetch)),)
