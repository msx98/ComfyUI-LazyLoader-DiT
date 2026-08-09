from .node import LazyLoadingDiTLoader
from .text_node import LazyQwenTextEncoderLoader

NODE_CLASS_MAPPINGS = {
    "LazyLoadingDiTLoader": LazyLoadingDiTLoader,
    # Key kept from the 8B-only release: it is what saved workflows reference.
    "LazyFluxQwen3_8BLoader": LazyQwenTextEncoderLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LazyLoadingDiTLoader": "Load Diffusion Model (Lazy DiT)",
    "LazyFluxQwen3_8BLoader": "Load Qwen Text Encoder (Lazy)",
}
