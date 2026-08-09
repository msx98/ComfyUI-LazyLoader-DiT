"""ComfyUI node declaration for lazy Qwen text encoders."""

import folder_paths

from .text_streaming import PROFILES, load_lazy_qwen_text_encoder


class LazyQwenTextEncoderLoader:
    """Lazy MPS loader for the single-file Qwen text encoders in PROFILES."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip_name": (folder_paths.get_filename_list("text_encoders"),),
            "memory_limit_mib": ("FLOAT", {"default": 6144.0, "min": 256.0, "max": 262144.0, "step": 256.0}),
            # New widgets are appended, never inserted: ComfyUI stores a saved
            # workflow's widget values positionally, so inserting one here would
            # shift every later value in workflows saved by the 8B-only release.
            "debug": ("BOOLEAN", {"default": False}),
            "profile": (["auto"] + sorted(PROFILES), {"default": "auto", "advanced": True}),
            "prefetch": ("BOOLEAN", {"default": True, "advanced": True}),
        }}

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "model/loaders"

    def load_clip(self, clip_name, memory_limit_mib=6144.0, debug=False, profile="auto", prefetch=True):
        path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        return (load_lazy_qwen_text_encoder(path, float(memory_limit_mib), bool(debug), str(profile), bool(prefetch)),)


# The old class name, for anything importing it directly.  The saved-workflow
# identifier is the NODE_CLASS_MAPPINGS key in __init__.py, not this name.
LazyFluxQwen3_8BLoader = LazyQwenTextEncoderLoader
