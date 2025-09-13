"""
SEAL-LOC: Semantic-Aware Localization Watermark

基于语义的定位水印嵌入器包，集成SEAL的语义感知能力与TAG-WM的双水印架构。
"""

from .seal_loc_embedder import SEALLOCEmbedder
from .simhash_utils import (
    simhash_single_patch, 
    simhash_batch_patches, 
    verify_simhash_consistency
)
from .patch_utils import (
    transform_img, 
    calculate_patch_l2, 
    calculate_patch_l2_all,
    visualize_patch_grid,
    map_latent_to_image_coords
)

__version__ = "1.0.0"
__author__ = "SEAL-LOC Team"
__description__ = "Semantic-Aware Localization Watermark for Diffusion Models"

__all__ = [
    'SEALLOCEmbedder',
    'simhash_single_patch',
    'simhash_batch_patches', 
    'verify_simhash_consistency',
    'transform_img',
    'calculate_patch_l2',
    'calculate_patch_l2_all',
    'visualize_patch_grid',
    'map_latent_to_image_coords'
] 