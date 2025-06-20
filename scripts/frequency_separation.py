"""
Frequency Separation Extension for Stable Diffusion WebUI reForge

This extension processes different frequency bands of an image independently through 
the diffusion process to overcome VAE limitations and achieve superior detail and quality.

Core concept:
- Split input image into frequency bands (low/mid/high) using FFT
- Process each band with optimal diffusion settings 
- Recombine bands for enhanced final result

Synchronization modes prevent ghosting and spatial misalignment:
- Independent: Original behavior (potential ghosting)
- Synchronized noise: Same random seed across bands
- Cross-attention sync: Low freq guides higher frequencies
- Progressive refinement: Sequential processing, each builds on previous
- Shared latent: Mixed processing with spatial guidance
"""

import math
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
from copy import deepcopy

from freq_sep.config import SyncMode, FrequencyBandConfig, FreqSepConfig

# Import version
import sys
import os
from modules import script_callbacks
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    from __version__ import __version__
except ImportError:
    __version__ = "1.0.1"  # Fallback version

try:
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    import gradio as gr
    
    import modules.scripts as scripts
    import modules.images as images
    import modules.processing as processing
    import modules.shared as shared
    from modules.processing import StableDiffusionProcessing, StableDiffusionProcessingImg2Img
    from modules import devices, sd_models
    
    # Try to import DiagonalGaussianDistribution with fallbacks for different webui versions
    DiagonalGaussianDistribution = None
    try:
        from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
    except ImportError:
        try:
            from sgm.modules.distributions import DiagonalGaussianDistribution
        except ImportError:
            try:
                from modules.distributions import DiagonalGaussianDistribution
            except ImportError:
                pass
    
    # Optional imports
    try:
        import cv2
    except ImportError:
        cv2 = None
        print("⚠️ OpenCV not available, using PIL for image operations")
        
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Frequency Separation: Missing dependencies: {e}")
    DEPENDENCIES_AVAILABLE = False
    
    # Create dummy classes to prevent import errors
    class DummyScript:
        def title(self): return "Frequency Separation (Dependencies Missing)"
        def show(self, is_img2img): return False
        def ui(self, is_img2img): return []
        def process(self, *args, **kwargs): pass

class FrequencyProcessor:
    """Core frequency processing utilities"""
    
    @staticmethod
    def image_to_frequency_domain(image: torch.Tensor) -> torch.Tensor:
        """Convert image to frequency domain using FFT"""
        # Ensure image is in the right format (B, C, H, W)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Apply FFT to each channel
        freq_domain = torch.fft.fftn(image, dim=(-2, -1))
        return freq_domain
    
    @staticmethod
    def frequency_to_image_domain(freq_data: torch.Tensor) -> torch.Tensor:
        """Convert frequency domain data back to image"""
        # Apply inverse FFT
        image = torch.fft.ifftn(freq_data, dim=(-2, -1)).real
        return image
    
    @staticmethod
    def create_frequency_mask(shape: Tuple[int, int], freq_range: Tuple[float, float], 
                            overlap_factor: float = 0.1) -> torch.Tensor:
        """Create a frequency domain mask for a specific band"""
        h, w = shape
        center_h, center_w = h // 2, w // 2
        
        # Create coordinate grids
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y = y.float() - center_h
        x = x.float() - center_w
        
        # Calculate normalized frequencies (0-1 range)
        max_freq = math.sqrt(center_h**2 + center_w**2)
        freq_magnitude = torch.sqrt(x**2 + y**2) / max_freq
        
        low_freq, high_freq = freq_range
        
        # Create soft mask with overlap
        mask = torch.zeros_like(freq_magnitude)
        
        # Soft transitions to prevent ringing artifacts
        transition_width = overlap_factor
        
        # Lower transition
        lower_transition = (freq_magnitude - low_freq) / transition_width
        lower_mask = torch.sigmoid(lower_transition * 10)  # Sharp sigmoid
        
        # Upper transition  
        upper_transition = (high_freq - freq_magnitude) / transition_width
        upper_mask = torch.sigmoid(upper_transition * 10)
        
        # Combine transitions
        mask = lower_mask * upper_mask
        
        return mask.to(freq_magnitude.device)
    
    @staticmethod
    def extract_frequency_band(freq_data: torch.Tensor, freq_range: Tuple[float, float],
                             overlap_factor: float = 0.1) -> torch.Tensor:
        """Extract a specific frequency band from frequency domain data"""
        shape = freq_data.shape[-2:]
        mask = FrequencyProcessor.create_frequency_mask(shape, freq_range, overlap_factor)
        
        # Apply mask to all channels and batches
        masked_freq = freq_data * mask.unsqueeze(0).unsqueeze(0)
        return masked_freq
    
    @staticmethod
    def split_into_frequency_bands(image: torch.Tensor, 
                                 config: FreqSepConfig) -> Dict[str, torch.Tensor]:
        """Split image into multiple frequency bands"""
        freq_domain = FrequencyProcessor.image_to_frequency_domain(image)
        
        bands = {}
        for band_config in config.band_configs:
            band_freq = FrequencyProcessor.extract_frequency_band(
                freq_domain, band_config.frequency_range, config.overlap_factor
            )
            band_image = FrequencyProcessor.frequency_to_image_domain(band_freq)
            bands[band_config.name] = band_image
        
        return bands
    
    @staticmethod
    def recombine_frequency_bands(bands: Dict[str, torch.Tensor],
                                config: FreqSepConfig) -> torch.Tensor:
        """Recombine processed frequency bands into final image"""
        # Convert each band back to frequency domain
        combined_freq = None
        
        for band_name, band_image in bands.items():
            band_config = next(cfg for cfg in config.band_configs if cfg.name == band_name)
            
            # Convert to frequency domain
            band_freq = FrequencyProcessor.image_to_frequency_domain(band_image)
            
            # Apply frequency mask to ensure proper band separation
            masked_freq = FrequencyProcessor.extract_frequency_band(
                band_freq, band_config.frequency_range, config.overlap_factor
            )
            
            if combined_freq is None:
                combined_freq = masked_freq
            else:
                combined_freq += masked_freq
        
        # Convert back to image domain
        final_image = FrequencyProcessor.frequency_to_image_domain(combined_freq)
        return final_image

class SynchronizationManager:
    """Handles different synchronization modes between frequency bands"""
    
    @staticmethod
    def prepare_synchronized_processing(bands: Dict[str, torch.Tensor], 
                                      config: FreqSepConfig,
                                      base_p: StableDiffusionProcessing) -> Dict[str, StableDiffusionProcessing]:
        """Prepare processing objects for each band based on sync mode"""
        processing_configs = {}
        
        if config.sync_mode == SyncMode.INDEPENDENT:
            # Independent processing - each band gets its own random seed
            for band_name, band_image in bands.items():
                p_copy = copy_processing_params(base_p)
                processing_configs[band_name] = p_copy
                
        elif config.sync_mode == SyncMode.SYNCHRONIZED_NOISE:
            # Same random seed for all bands
            shared_seed = base_p.seed if base_p.seed != -1 else torch.randint(0, 2**32, (1,)).item()
            for band_name, band_image in bands.items():
                p_copy = copy_processing_params(base_p)
                p_copy.seed = shared_seed
                processing_configs[band_name] = p_copy
                
        elif config.sync_mode == SyncMode.CROSS_ATTENTION:
            # Low frequency guides higher frequencies
            for band_name, band_image in bands.items():
                p_copy = copy_processing_params(base_p)
                # Implementation will be extended in process_with_cross_attention
                processing_configs[band_name] = p_copy
                
        elif config.sync_mode == SyncMode.PROGRESSIVE:
            # Sequential processing setup
            for band_name, band_image in bands.items():
                p_copy = copy_processing_params(base_p)
                processing_configs[band_name] = p_copy
                
        elif config.sync_mode == SyncMode.SHARED_LATENT:
            # Shared latent space processing
            for band_name, band_image in bands.items():
                p_copy = copy_processing_params(base_p)
                processing_configs[band_name] = p_copy
        
        return processing_configs
    
    @staticmethod
    def process_bands_with_sync(bands: Dict[str, torch.Tensor],
                              processing_configs: Dict[str, StableDiffusionProcessing],
                              config: FreqSepConfig) -> Dict[str, torch.Tensor]:
        """Process all frequency bands according to the synchronization mode"""
        
        if config.sync_mode == SyncMode.INDEPENDENT:
            return SynchronizationManager._process_independent(bands, processing_configs, config)
        elif config.sync_mode == SyncMode.SYNCHRONIZED_NOISE:
            return SynchronizationManager._process_synchronized_noise(bands, processing_configs, config)
        elif config.sync_mode == SyncMode.CROSS_ATTENTION:
            return SynchronizationManager._process_cross_attention(bands, processing_configs, config)
        elif config.sync_mode == SyncMode.PROGRESSIVE:
            return SynchronizationManager._process_progressive(bands, processing_configs, config)
        elif config.sync_mode == SyncMode.SHARED_LATENT:
            return SynchronizationManager._process_shared_latent(bands, processing_configs, config)
    
    @staticmethod
    def _process_independent(bands: Dict[str, torch.Tensor],
                           processing_configs: Dict[str, StableDiffusionProcessing],
                           config: FreqSepConfig) -> Dict[str, torch.Tensor]:
        """Process each band independently"""
        processed_bands = {}
        
        for band_name, band_image in bands.items():
            band_config = next(cfg for cfg in config.band_configs if cfg.name == band_name)
            p = processing_configs[band_name]
            
            # Apply band-specific settings
            p.denoising_strength = band_config.denoising_strength
            p.steps = band_config.steps
            p.cfg_scale = band_config.cfg_scale
            
            # Convert tensor to PIL for processing
            pil_image = tensor_to_pil(band_image)
            p.init_images = [pil_image]
            
            # Process the band
            processed = processing.process_images(p)
            
            # Convert back to tensor
            processed_tensor = pil_to_tensor(processed.images[0])
            processed_bands[band_name] = processed_tensor
        
        return processed_bands
    
    @staticmethod
    def _process_synchronized_noise(bands: Dict[str, torch.Tensor],
                                  processing_configs: Dict[str, StableDiffusionProcessing],
                                  config: FreqSepConfig) -> Dict[str, torch.Tensor]:
        """Process with synchronized noise across all bands"""
        # Same as independent but with shared seeds (already set in prepare_synchronized_processing)
        return SynchronizationManager._process_independent(bands, processing_configs, config)
    
    @staticmethod
    def _process_cross_attention(bands: Dict[str, torch.Tensor],
                               processing_configs: Dict[str, StableDiffusionProcessing], 
                               config: FreqSepConfig) -> Dict[str, torch.Tensor]:
        """Process with cross-attention synchronization"""
        processed_bands = {}
        low_freq_result = None
        
        # Sort bands by frequency (low to high)
        sorted_bands = sorted(bands.items(), 
                            key=lambda x: next(cfg.frequency_range[0] for cfg in config.band_configs if cfg.name == x[0]))
        
        for band_name, band_image in sorted_bands:
            band_config = next(cfg for cfg in config.band_configs if cfg.name == band_name)
            p = processing_configs[band_name]
            
            # Apply band-specific settings
            p.denoising_strength = band_config.denoising_strength
            p.steps = band_config.steps
            p.cfg_scale = band_config.cfg_scale
            
            if low_freq_result is not None and not band_config.preserve_composition:
                # Use low frequency result as spatial guidance
                guidance_strength = config.spatial_guidance_strength
                
                # Blend with low frequency guidance
                guided_image = blend_with_guidance(band_image, low_freq_result, guidance_strength)
                pil_image = tensor_to_pil(guided_image)
            else:
                pil_image = tensor_to_pil(band_image)
            
            p.init_images = [pil_image]
            
            # Process the band
            processed = processing.process_images(p)
            processed_tensor = pil_to_tensor(processed.images[0])
            processed_bands[band_name] = processed_tensor
            
            # Store low frequency result for guidance
            if band_config.preserve_composition:
                low_freq_result = processed_tensor
        
        return processed_bands
    
    @staticmethod
    def _process_progressive(bands: Dict[str, torch.Tensor],
                           processing_configs: Dict[str, StableDiffusionProcessing],
                           config: FreqSepConfig) -> Dict[str, torch.Tensor]:
        """Process bands progressively, each building on the previous"""
        processed_bands = {}
        cumulative_result = None
        
        # Sort bands by frequency (low to high)
        sorted_bands = sorted(bands.items(),
                            key=lambda x: next(cfg.frequency_range[0] for cfg in config.band_configs if cfg.name == x[0]))
        
        for i, (band_name, band_image) in enumerate(sorted_bands):
            band_config = next(cfg for cfg in config.band_configs if cfg.name == band_name)
            p = processing_configs[band_name]
            
            # Apply band-specific settings
            p.denoising_strength = band_config.denoising_strength
            p.steps = band_config.steps
            p.cfg_scale = band_config.cfg_scale
            
            if cumulative_result is not None:
                # Combine current band with cumulative result
                combined_image = combine_progressive_bands(band_image, cumulative_result, config)
                pil_image = tensor_to_pil(combined_image)
            else:
                pil_image = tensor_to_pil(band_image)
            
            p.init_images = [pil_image]
            
            # Process the band
            processed = processing.process_images(p)
            processed_tensor = pil_to_tensor(processed.images[0])
            processed_bands[band_name] = processed_tensor
            
            # Update cumulative result
            if cumulative_result is None:
                cumulative_result = processed_tensor
            else:
                cumulative_result = update_cumulative_result(cumulative_result, processed_tensor, band_config)
        
        return processed_bands
    
    @staticmethod  
    def _process_shared_latent(bands: Dict[str, torch.Tensor],
                             processing_configs: Dict[str, StableDiffusionProcessing],
                             config: FreqSepConfig) -> Dict[str, torch.Tensor]:
        """Process with shared latent space"""
        processed_bands = {}
        
        # Create shared latent representation
        combined_input = torch.stack(list(bands.values())).mean(dim=0)
        
        for band_name, band_image in bands.items():
            band_config = next(cfg for cfg in config.band_configs if cfg.name == band_name)
            p = processing_configs[band_name]
            
            # Apply band-specific settings
            p.denoising_strength = band_config.denoising_strength
            p.steps = band_config.steps
            p.cfg_scale = band_config.cfg_scale
            
            # Mix with shared representation
            guidance_strength = config.spatial_guidance_strength
            mixed_image = (1 - guidance_strength) * band_image + guidance_strength * combined_input
            
            pil_image = tensor_to_pil(mixed_image)
            p.init_images = [pil_image]
            
            # Process the band
            processed = processing.process_images(p)
            processed_tensor = pil_to_tensor(processed.images[0])
            processed_bands[band_name] = processed_tensor
        
        return processed_bands


# Utility functions
def copy_processing_params(p: StableDiffusionProcessing) -> StableDiffusionProcessing:
    """Create a copy of processing parameters - simplified version"""
    try:
        # Use deepcopy for safety - this is more reliable than manual copying
        p_copy = deepcopy(p)
        return p_copy
    except Exception as e:
        print(f"⚠️ Error copying processing params: {e}")
        return p


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Ensure tensor is in CPU and correct format
    tensor = tensor.detach().cpu()
    
    # Convert from [-1, 1] to [0, 1] if needed
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    
    # Clamp values
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to [0, 255] and proper format
    tensor = (tensor * 255).byte()
    
    # Convert to numpy and rearrange dimensions
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)
    
    numpy_array = tensor.numpy()
    
    return Image.fromarray(numpy_array)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL Image to tensor"""
    # Convert to numpy
    numpy_array = np.array(image)
    
    # Convert to torch tensor
    tensor = torch.from_numpy(numpy_array).float() / 255.0
    
    # Rearrange dimensions to (C, H, W)
    if tensor.dim() == 3:
        tensor = tensor.permute(2, 0, 1)
    
    # Convert to [-1, 1] range
    tensor = tensor * 2 - 1
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor


def blend_with_guidance(target: torch.Tensor, guidance: torch.Tensor, strength: float) -> torch.Tensor:
    """Blend target image with guidance image"""
    return (1 - strength) * target + strength * guidance


def combine_progressive_bands(current_band: torch.Tensor, cumulative: torch.Tensor, 
                            config: FreqSepConfig) -> torch.Tensor:
    """Combine current frequency band with cumulative result for progressive processing"""
    # Simple additive combination - can be enhanced with more sophisticated blending
    return current_band + cumulative * 0.3


def update_cumulative_result(cumulative: torch.Tensor, new_result: torch.Tensor,
                           band_config: FrequencyBandConfig) -> torch.Tensor:
    """Update cumulative result with new processed band"""
    # Weight based on frequency range importance
    weight = 0.7 if band_config.preserve_composition else 0.5
    return weight * cumulative + (1 - weight) * new_result


class FrequencySeparationScript(scripts.Script):
    """Main script class for the frequency separation extension"""
    
    def ui(self, is_img2img):
        """UI method required by scripts.Script interface"""
        from freq_sep.ui import ui as create_ui
        ui_components, self.infotext_fields = create_ui(is_img2img)
        return ui_components

    def __init__(self):
        super().__init__()
        print("🌊 Frequency Separation Extension: Initialized successfully!")
        # Set priority to run before ADetailer
        # Lower numbers run first. ADetailer typically doesn't set priority (defaults to 0)
        self.priority = -10
        
        # Debug configuration
        self.debug_mode = True  # Set to True to enable debug outputs
        self.debug_folder = None
    
    def _setup_debug_folder(self, p):
        """Create debug folder if debug mode is enabled"""
        if self.debug_mode and self.debug_folder is None:
            import datetime
            import os
            from modules.shared import opts
            
            # Determine if we're in txt2img or img2img mode
            if p is not None:
                # Check the class type to determine original mode, not just presence of init_images
                is_img2img = 'Img2Img' in type(p).__name__
                print(f"  🐛 Debug: Processing type is {type(p).__name__}, is_img2img={is_img2img}")
                mode = "img2img" if is_img2img else "txt2img"
                
                # Get the appropriate output directory
                if is_img2img:
                    base_dir = opts.outdir_img2img_samples or "outputs/img2img-images"
                else:
                    base_dir = opts.outdir_txt2img_samples or "outputs/txt2img-images"
            else:
                # Fallback if no processing object available
                base_dir = "outputs"
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.debug_folder = os.path.join(base_dir, f"frequency_separation_debug_{timestamp}")
            os.makedirs(self.debug_folder, exist_ok=True)
            print(f"🐛 Debug mode enabled - saving debug images to {self.debug_folder}/")
    
    def _save_debug_image(self, image_array, filename, p, stage=""):
        """Save debug image if debug mode is enabled"""
        if not self.debug_mode:
            return
        
        self._setup_debug_folder(p)
        filepath = os.path.join(self.debug_folder, filename)
        Image.fromarray(image_array).save(filepath)
        print(f"  💾 [{stage}] Saved debug image: {filename}")
    
    def title(self):
        return "Frequency Separation Enhancement"
    
    def show(self, is_img2img):
        return scripts.AlwaysVisible  # Always visible as dedicated section
    
    def process(self, p: StableDiffusionProcessing, enabled: bool, sync_mode: str, num_bands: int,
               overlap_factor: float, spatial_guidance: float, recombination_method: str,
               save_before_denoising: bool, use_custom_steps_cfg: bool, preserve_dc_component_v2: bool, use_fft_shift: bool, use_correct_fft_shift: bool, mask_function: str,
               low_freq_start: float, low_freq_end: float, low_denoising: float, low_amplitude: float, low_steps: int, low_cfg: float,
               mid_freq_start: float, mid_freq_end: float, mid_denoising: float, mid_amplitude: float, mid_steps: int, mid_cfg: float,
               high_freq_start: float, high_freq_end: float, high_denoising: float, high_amplitude: float, high_steps: int, high_cfg: float):
        """Main processing method - works on latents, not final images"""
        
        print(f"🔍 RAW ARGS DEBUG: preserve_dc_component_v2={preserve_dc_component_v2}, use_correct_fft_shift={use_correct_fft_shift}, mask_function={mask_function}")
        print(f"🔍 Frequency Separation: process() called - enabled={enabled}, p_type={type(p).__name__}")
        
        # Skip if not enabled
        if not enabled:
            print("🔄 Frequency Separation: Disabled by user, skipping processing")
            return
            
        # Create frequency separation configuration first (needed for both modes)
        # Use custom steps/cfg only if checkbox is enabled
        band_configs = [
            FrequencyBandConfig(
                name="low_freq",
                frequency_range=(low_freq_start, low_freq_end),
                denoising_strength=low_denoising,
                steps=low_steps if use_custom_steps_cfg else p.steps,
                cfg_scale=low_cfg if use_custom_steps_cfg else p.cfg_scale,
                amplitude_scale=low_amplitude,
                preserve_composition=True
            ),
            FrequencyBandConfig(
                name="mid_freq", 
                frequency_range=(mid_freq_start, mid_freq_end),
                denoising_strength=mid_denoising,
                steps=mid_steps if use_custom_steps_cfg else p.steps,
                cfg_scale=mid_cfg if use_custom_steps_cfg else p.cfg_scale,
                amplitude_scale=mid_amplitude
            ),
            FrequencyBandConfig(
                name="high_freq",
                frequency_range=(high_freq_start, high_freq_end),
                denoising_strength=high_denoising,
                steps=high_steps if use_custom_steps_cfg else p.steps,
                cfg_scale=high_cfg if use_custom_steps_cfg else p.cfg_scale,
                amplitude_scale=high_amplitude
            )
        ]
        
        # Handle sync_mode conversion safely
        try:
            if isinstance(sync_mode, str):
                sync_mode_enum = SyncMode(sync_mode)
            else:
                # Handle case where it might be a tuple (display_name, value)
                if isinstance(sync_mode, (list, tuple)) and len(sync_mode) == 2:
                    sync_mode_enum = SyncMode(sync_mode[1])
                else:
                    sync_mode_enum = SyncMode(str(sync_mode))
        except ValueError as e:
            print(f"⚠️ Invalid sync mode '{sync_mode}', using progressive refinement")
            sync_mode_enum = SyncMode.PROGRESSIVE
        
        freq_config = FreqSepConfig(
            sync_mode=sync_mode_enum,
            num_bands=num_bands,
            band_configs=band_configs[:num_bands],
            overlap_factor=overlap_factor,
            spatial_guidance_strength=spatial_guidance
        )
        
        # Store all parameters on p for metadata generation (needed for both modes)
        p._freq_sep_enabled = True
        p._freq_sep_config = freq_config
        p._freq_sep_recombination_method = recombination_method
        p._freq_sep_save_before_denoising = save_before_denoising
        p._freq_sep_use_custom_steps_cfg = use_custom_steps_cfg
        p._freq_sep_preserve_dc_component = preserve_dc_component_v2
        p._freq_sep_use_fft_shift = use_fft_shift
        p._freq_sep_use_correct_fft_shift = use_correct_fft_shift
        p._freq_sep_mask_function = mask_function
        
        # Check if we're in txt2img or img2img mode
        is_txt2img = not isinstance(p, StableDiffusionProcessingImg2Img)
        
        if is_txt2img:
            print("🎨 Frequency Separation: txt2img mode detected - will enhance after generation")
            print(f"   📊 Using custom steps/cfg: {use_custom_steps_cfg}")
            if use_custom_steps_cfg:
                print(f"   📊 Custom band steps: low={low_steps}, mid={mid_steps}, high={high_steps}")
                print(f"   📊 Custom band cfg: low={low_cfg}, mid={mid_cfg}, high={high_cfg}")
            else:
                print(f"   📊 Using generation settings: steps={p.steps}, cfg={p.cfg_scale}")
            
            # Add metadata to PNG parameters
            extra_params = self.extra_params(p)
            p.extra_generation_params.update(extra_params)
            
            return  # Let normal txt2img generation proceed
        
        # img2img mode checks
        if not p.init_images:
            print("⚠️ Frequency Separation: No input images provided, skipping")
            return
            
        # Prevent recursion when processing frequency bands
        if hasattr(p, '_frequency_separation_disabled'):
            print("🔄 Frequency Separation: Disabled for internal frequency band processing, skipping")
            return
            
        # Debug: Check what flags are present on the processing object
        flags = []
        if hasattr(p, '_ad_disabled'):
            flags.append(f"_ad_disabled={getattr(p, '_ad_disabled')}")
        if hasattr(p, '_ad_inner'):
            flags.append(f"_ad_inner={getattr(p, '_ad_inner')}")
        if hasattr(p, '_frequency_separation_disabled'):
            flags.append(f"_frequency_separation_disabled={getattr(p, '_frequency_separation_disabled')}")
        
        if flags:
            print(f"🔍 Frequency Separation: Processing object flags: {', '.join(flags)}")
        else:
            print("🔍 Frequency Separation: No extension flags detected on processing object")
            
        print("🌊 Frequency Separation: Starting latent-space frequency processing...")
        
        try:
            print("🎆 STARTING FREQUENCY SEPARATION WITH REAL DIFFUSION PROCESSING")
            print(f"   ⚙️ Mode: {freq_config.sync_mode.value}")
            print(f"   📋 Bands: {len(freq_config.band_configs)}")
            print(f"   🎯 This will run ACTUAL WebUI diffusion on each frequency band with progress bars!")
            print(f"   🔍 Using same approach as ADetailer for reliable processing")
            
            # Process in latent space and modify p.init_images in place
            enhanced_images = self.process_latent_frequency_separation(p, freq_config, recombination_method, save_before_denoising, preserve_dc_component_v2, use_fft_shift, use_correct_fft_shift, mask_function)
            
            # CRITICAL: Replace the init_images so img2img uses our enhanced versions
            if enhanced_images:
                p.init_images = enhanced_images
                print(f"🔄 Replaced {len(enhanced_images)} init images with frequency-enhanced versions")
                print("🎯 img2img will now process the enhanced images through normal diffusion")
                print("🎉 FREQUENCY SEPARATION COMPLETE - Enhanced images ready for final img2img!")
                print(f"   📈 Total diffusion operations: {len(freq_config.band_configs)} bands per image")
                print(f"   ⏱️ Each band was processed through full WebUI img2img pipeline")
            else:
                print("⚠️ No enhanced images produced, using original images")
            
            # For img2img, we replace init_images and let normal processing continue
            # But if save_before_denoising is True, we actually want to output these directly!
            if save_before_denoising:
                print("🎯 Outputting frequency-enhanced images directly (skipping final denoising)")
                # Add metadata to PNG parameters for direct output
                extra_params = self.extra_params(p)
                p.extra_generation_params.update(extra_params)
                
                # Create a Processed object with our enhanced images
                from modules.processing import Processed
                processed = Processed(
                    p,
                    images_list=enhanced_images,
                    seed=p.seed,
                    info=p.info if hasattr(p, 'info') else ""
                )
                return processed
            else:
                # Normal behavior - continue with img2img denoising
                # Add metadata to PNG parameters for final output
                extra_params = self.extra_params(p)
                p.extra_generation_params.update(extra_params)
                return None
            
        except Exception as e:
            print(f"❌ Error in latent frequency separation: {e}")
            import traceback
            traceback.print_exc()
            print("🔄 Falling back to normal processing")
            return None
    
    def process_latent_frequency_separation(self, p: StableDiffusionProcessingImg2Img, freq_config: FreqSepConfig, recombination_method: str, save_before_denoising: bool, preserve_dc_component: bool = False, use_fft_shift: float = 1.0, use_correct_fft_shift: bool = False, mask_function: str = "center_circular"):
        """Core latent-space frequency separation processing WITH REAL DIFFUSION"""
        
        print(f"🔧 Using recombination method: {recombination_method}")
        print(f"💾 Save before denoising: {save_before_denoising}")
        print(f"💡 Preserve DC component: {preserve_dc_component}")
        print(f"🔧 FFT Shift: {use_fft_shift}")
        print(f"🎯 FFT Shift Detail Inspection: {use_correct_fft_shift}")
        print(f"🎭 Mask Function: {mask_function}")
        print(f"🔍 DEBUG: preserve_dc_component={preserve_dc_component}, use_fft_shift={use_fft_shift}, use_correct_fft_shift={use_correct_fft_shift}, mask_function={mask_function}")
        
        print(f"🧠 Processing {len(p.init_images)} images with REAL DIFFUSION in latent space")
        print(f"   🌊 Sync Mode: {freq_config.sync_mode.value}")
        print(f"   🖼️ Frequency Bands: {len(freq_config.band_configs)}")
        
        band_info = []
        for band in freq_config.band_configs:
            band_info.append(f"{band.name}({band.frequency_range[0]:.2f}-{band.frequency_range[1]:.2f}, denoise:{band.denoising_strength:.2f}, steps:{band.steps})")
        print(f"   🏢 Band Configuration: {', '.join(band_info)}")
        
        # Import device utilities
        from modules import devices
        
        final_images = []
        
        for i, init_image in enumerate(p.init_images):
            print(f"\n🎯 ==> Processing image {i+1}/{len(p.init_images)} with REAL DIFFUSION...")
            
            # 1. Encode image to latent space
            print(f"  🔄 Step 1: Encoding image to latent space...")
            init_latent = self.encode_image_to_latent(init_image, p)
            init_mean = torch.mean(init_latent).item()
            init_std = torch.std(init_latent).item()
            init_min = torch.min(init_latent).item()
            init_max = torch.max(init_latent).item()
            
            # Capture input latent energy for normalization reference
            input_latent_energy = torch.mean(init_latent ** 2).item()
            print(f"     ✅ Encoded to latent shape: {init_latent.shape}")
            print(f"     📊 Original latent stats: mean={init_mean:.3f}, std={init_std:.3f}, range=[{init_min:.3f}, {init_max:.3f}]")
            print(f"     🔋 Input latent energy: {input_latent_energy:.3f} (reference for normalization)")
            
            # 2. Split latent into frequency bands using FFT
            print(f"  🌊 Step 2: Splitting into frequency bands...")
            frequency_bands = self.split_latent_frequency_bands(init_latent, freq_config, preserve_dc_component, use_fft_shift, use_correct_fft_shift, mask_function, p)
            print(f"     ✅ Split into {len(frequency_bands)} frequency bands")
            
            # 3. Process each frequency band with ACTUAL DIFFUSION
            print(f"  🔥 Step 3: Running REAL DIFFUSION on each frequency band...")
            print(f"     ⚠️ This will take time and show progress bars for each band!")
            processed_bands = self.process_frequency_bands_in_latent_space(
                frequency_bands, freq_config, p, i
            )
            
            # Debug: Check latent stats after processing
            total_processed_mean = 0
            total_processed_std = 0
            for band_name, band_latent in processed_bands.items():
                band_mean = torch.mean(band_latent).item()
                band_std = torch.std(band_latent).item()
                total_processed_mean += band_mean
                total_processed_std += band_std
                print(f"     📊 {band_name} processed latent: mean={band_mean:.3f}, std={band_std:.3f}")
            
            avg_processed_mean = total_processed_mean / len(processed_bands)
            avg_processed_std = total_processed_std / len(processed_bands)
            print(f"     📊 Average processed latent: mean={avg_processed_mean:.3f}, std={avg_processed_std:.3f}")
            print(f"     🔍 Mean change: {init_mean:.3f} → {avg_processed_mean:.3f} (diff: {avg_processed_mean - init_mean:.3f})")
            print(f"     🔍 Std change: {init_std:.3f} → {avg_processed_std:.3f} (diff: {avg_processed_std - init_std:.3f})")
            
            # 3.5. ENERGY NORMALIZATION: Normalize processed latents to match input energy
            print(f"  ⚡ Step 3.5: Latent space energy normalization...")
            
            # Calculate total energy of processed latent bands
            total_processed_latent_energy = 0
            for band_name, band_latent in processed_bands.items():
                band_energy = torch.mean(band_latent ** 2).item()
                total_processed_latent_energy += band_energy
                print(f"     🔋 {band_name} latent energy: {band_energy:.3f}")
            
            print(f"     🔋 Total processed latent energy: {total_processed_latent_energy:.3f}")
            print(f"     🔋 Input latent energy (reference): {input_latent_energy:.3f}")
            
            # Apply energy normalization to preserve brightness and saturation
            if total_processed_latent_energy > 0:
                energy_ratio = math.sqrt(input_latent_energy / total_processed_latent_energy)
                print(f"     ⚡ Energy correction factor: {energy_ratio:.3f}")
                
                # Apply energy correction to each band
                for band_name in processed_bands:
                    processed_bands[band_name] = processed_bands[band_name] * energy_ratio
                    corrected_energy = torch.mean(processed_bands[band_name] ** 2).item()
                    print(f"     ✅ {band_name} corrected energy: {corrected_energy:.3f}")
                
                # Verify total energy after correction
                total_corrected_energy = sum(torch.mean(band ** 2).item() for band in processed_bands.values())
                print(f"     🔋 Total corrected energy: {total_corrected_energy:.3f} (target: {input_latent_energy:.3f})")
            else:
                print(f"     ⚠️ Skipping energy normalization (total processed energy is zero)")
            
            # Normalize latents to preserve original distribution before VAE decoding
            print(f"  🎯 Step 3.6: Normalizing latent distribution to preserve saturation...")
            
            # Calculate current combined statistics
            all_processed_latents = torch.cat([band.flatten() for band in processed_bands.values()])
            current_mean = torch.mean(all_processed_latents).item()
            current_std = torch.std(all_processed_latents).item()
            
            print(f"     📊 Current combined latent: mean={current_mean:.3f}, std={current_std:.3f}")
            print(f"     📊 Target (original) latent: mean={init_mean:.3f}, std={init_std:.3f}")
            
            # Apply distribution matching to preserve saturation
            if current_std > 0:
                for band_name in processed_bands:
                    # Normalize to match original distribution
                    band_latent = processed_bands[band_name]
                    band_latent = (band_latent - current_mean) / current_std  # Standardize
                    band_latent = band_latent * init_std + init_mean  # Match original distribution
                    processed_bands[band_name] = band_latent
                    
                    # Verify the correction
                    band_mean = torch.mean(band_latent).item()
                    band_std = torch.std(band_latent).item()
                    print(f"     ✅ {band_name} normalized: mean={band_mean:.3f}, std={band_std:.3f}")
                
                # Verify combined statistics after normalization
                all_normalized = torch.cat([band.flatten() for band in processed_bands.values()])
                final_mean = torch.mean(all_normalized).item()
                final_std = torch.std(all_normalized).item()
                print(f"     🎯 Final combined latent: mean={final_mean:.3f}, std={final_std:.3f}")
            
            print(f"  🎨 Step 4: Multi-VAE decoding each frequency band separately...")

            decoded_bands = {}
            band_brightnesses = {}
            for band_name, band_latent in processed_bands.items():
                print(f"    🖼️ VAE decoding {band_name} band...")
                decoded_band_image = self.decode_latent_to_image(band_latent, p)
                decoded_bands[band_name] = decoded_band_image
                
                # Track brightness of each decoded band
                band_array = np.array(decoded_band_image)
                band_brightness = np.mean(band_array)
                band_brightnesses[band_name] = band_brightness
                print(f"      ✅ {band_name} band decoded: {decoded_band_image.size}, brightness: {band_brightness:.1f}")
            
            # Calculate target brightness (average of all bands weighted by their importance)
            target_brightness = np.mean(list(band_brightnesses.values()))
            print(f"    🎯 Target brightness for normalization: {target_brightness:.1f}")
            
            # Normalize brightness of each band to match target
            for band_name, band_image in decoded_bands.items():
                if band_brightnesses[band_name] > 0:
                    brightness_ratio = target_brightness / band_brightnesses[band_name]
                    if abs(brightness_ratio - 1.0) > 0.05:  # Only adjust if significant difference
                        print(f"    🔆 Adjusting {band_name} brightness by factor {brightness_ratio:.3f}")
                        band_array = np.array(band_image).astype(np.float32)
                        band_array = band_array * brightness_ratio
                        band_array = np.clip(band_array, 0, 255).astype(np.uint8)
                        decoded_bands[band_name] = Image.fromarray(band_array, 'RGB')
            
            # # 5. Recombine the multiple VAE-decoded images in image space
            # print(f"  🔥 Step 5: Recombining multiple VAE renderings in image space...")
            
            # # Debug: Check brightness of decoded bands before recombination
            # for band_name, band_image in decoded_bands.items():
            #     band_array = np.array(band_image)
            #     band_brightness = np.mean(band_array)
            #     print(f"     💡 {band_name} band brightness: {band_brightness:.1f}")
            
            final_image = self.recombine_decoded_frequency_bands(decoded_bands, freq_config, recombination_method, mask_function)
            
            # # Debug: Check final image brightness after recombination
            # final_array = np.array(final_image)
            # final_brightness = np.mean(final_array)
            # original_brightness = np.mean(np.array(init_image))
            # print(f"     💡 Original image brightness: {original_brightness:.1f}")
            # print(f"     💡 Final image brightness: {final_brightness:.1f}")
            # print(f"     🔍 Brightness change: {original_brightness:.1f} → {final_brightness:.1f} (diff: {final_brightness - original_brightness:.1f})")
            
            # print(f"     ✅ Multi-VAE recombination complete: {final_image.size}")
            
            # Save the frequency-enhanced image before final denoising (if enabled)
            if save_before_denoising:
                from modules import images
                import os
                
                # Create output directory if it doesn't exist
                output_dir = p.outpath_samples if hasattr(p, 'outpath_samples') else "outputs/txt2img-images"
                os.makedirs(output_dir, exist_ok=True)
            
            final_images.append(final_image)
            print(f"  🎆 Image {i+1}/{len(p.init_images)} frequency processing COMPLETE!")
        
        print(f"\n🎉 ALL FREQUENCY SEPARATION WITH REAL DIFFUSION COMPLETE!")
        print(f"   ✨ Enhanced {len(final_images)} images through frequency-domain processing")
        print(f"   🎯 Each image processed through {len(freq_config.band_configs)} separate diffusion runs")
        
        # Return the enhanced images for img2img to process further
        return final_images
    
    def encode_image_to_latent(self, image: Image.Image, p: StableDiffusionProcessingImg2Img) -> torch.Tensor:
        """Encode PIL image to latent tensor"""
        try:
            from modules import images
            from modules.sd_samplers_common import images_tensor_to_samples
            
            # Convert PIL to proper format for WebUI
            # Ensure RGB format (no alpha channel)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to match processing dimensions
            if hasattr(p, 'width') and hasattr(p, 'height'):
                image = image.resize((p.width, p.height), Image.LANCZOS)
            
            # Convert PIL to numpy then to tensor
            image_np = np.array(image).astype(np.float32) / 255.0
            
            # Ensure we have exactly 3 channels (RGB)
            if len(image_np.shape) == 2:  # Grayscale
                image_np = np.stack([image_np] * 3, axis=-1)
            elif image_np.shape[2] == 4:  # RGBA
                image_np = image_np[:, :, :3]  # Drop alpha channel
            elif image_np.shape[2] != 3:
                raise ValueError(f"Unexpected number of channels: {image_np.shape[2]}")
            
            # Convert to torch tensor (H, W, C) -> (C, H, W) -> (1, C, H, W)
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
            
            # Scale to [-1, 1] range as expected by VAE
            image_tensor = image_tensor * 2.0 - 1.0
            
            # Move to correct device
            device = devices.device
            dtype = devices.dtype_vae if hasattr(devices, 'dtype_vae') else torch.float32
            image_tensor = image_tensor.to(device=device, dtype=dtype)
            
            # Encode using model's VAE encoder
            with torch.no_grad():
                if hasattr(p.sd_model, 'encode_first_stage'):
                    latent = p.sd_model.encode_first_stage(image_tensor)
                elif hasattr(p.sd_model, 'first_stage_model'):
                    latent = p.sd_model.first_stage_model.encode(image_tensor)
                else:
                    raise Exception("No VAE encoder method found")

                # Handle DiagonalGaussianDistribution if available
                if DiagonalGaussianDistribution is not None and isinstance(latent, DiagonalGaussianDistribution):
                    latent = latent.parameters
                elif hasattr(latent, 'parameters'):
                    # Fallback for objects that might have parameters attribute
                    latent = latent.parameters

                # Apply scaling factor if present
                if hasattr(p.sd_model, 'scale_factor'):
                    latent = latent * p.sd_model.scale_factor
                elif hasattr(p.sd_model, 'first_stage_model') and hasattr(p.sd_model.first_stage_model, 'scale_factor'):
                    latent = latent * p.sd_model.first_stage_model.scale_factor
            
            print(f"    📐 Successfully encoded image to latent: {latent.shape}")
            return latent
            
        except Exception as e:
            print(f"⚠️ Error encoding image to latent: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: create appropriately sized dummy latent
            try:
                height = getattr(p, 'height', 512) // 8  # VAE typically downsamples by 8x
                width = getattr(p, 'width', 512) // 8
                device = devices.device
                dtype = devices.dtype_vae if hasattr(devices, 'dtype_vae') else torch.float32
                dummy_latent = torch.randn(1, 4, height, width, device=device, dtype=dtype)
                print(f"    🔄 Created fallback dummy latent: {dummy_latent.shape}")
                return dummy_latent
            except Exception as e2:
                print(f"⚠️ Error creating fallback latent: {e2}")
                # Ultimate fallback
                return torch.randn(1, 4, 64, 64, device=devices.device)
    
    def decode_latent_to_image(self, latent: torch.Tensor, p: StableDiffusionProcessingImg2Img) -> Image.Image:
        """Decode latent tensor to PIL image"""
        try:
            with torch.no_grad():
                # Unscale if needed
                latent_for_decode = latent.clone()
                if hasattr(p.sd_model, 'scale_factor'):
                    latent_for_decode = latent_for_decode / p.sd_model.scale_factor
                elif hasattr(p.sd_model, 'first_stage_model') and hasattr(p.sd_model.first_stage_model, 'scale_factor'):
                    latent_for_decode = latent_for_decode / p.sd_model.first_stage_model.scale_factor
                
                # Match dtype with VAE model to avoid dtype mismatch errors
                if hasattr(p.sd_model, 'first_stage_model') and p.sd_model.first_stage_model is not None:
                    try:
                        # Get the dtype from the VAE model's first parameter
                        vae_dtype = next(p.sd_model.first_stage_model.parameters()).dtype
                        if latent_for_decode.dtype != vae_dtype:
                            latent_for_decode = latent_for_decode.to(dtype=vae_dtype)
                    except (StopIteration, AttributeError):
                        # VAE model has no parameters or parameters() method doesn't exist
                        pass
                
                # Decode using VAE
                if hasattr(p.sd_model, 'decode_first_stage'):
                    decoded = p.sd_model.decode_first_stage(latent_for_decode)
                elif hasattr(p.sd_model, 'first_stage_model'):
                    decoded = p.sd_model.first_stage_model.decode(latent_for_decode)
                else:
                    raise Exception("No VAE decoder method found")
                
                # Convert tensor to PIL image
                decoded = torch.clamp(decoded, -1.0, 1.0)
                decoded = (decoded + 1.0) / 2.0  # Scale from [-1,1] to [0,1]
                # Convert BFloat16 to Float32 before numpy conversion if needed
                if decoded.dtype == torch.bfloat16:
                    decoded = decoded.float()
                decoded = decoded.squeeze(0).permute(1, 2, 0).cpu().numpy()
                decoded = (decoded * 255.0).astype(np.uint8)
                
                # Ensure we have RGB channels
                if decoded.shape[2] == 3:
                    image = Image.fromarray(decoded, 'RGB')
                else:
                    # Handle grayscale or other formats
                    image = Image.fromarray(decoded[:,:,0], 'L').convert('RGB')
                
                print(f"    🖼️ Successfully decoded latent to image: {image.size}")
                return image
                
        except Exception as e:
            print(f"⚠️ Error decoding latent to image: {e}")
            import traceback
            traceback.print_exc()
            
            # Return appropriately sized fallback image
            try:
                width = getattr(p, 'width', 512)
                height = getattr(p, 'height', 512)
                fallback_image = Image.new('RGB', (width, height), color='black')
                print(f"    🔄 Created fallback image: {fallback_image.size}")
                return fallback_image
            except Exception as e2:
                print(f"⚠️ Error creating fallback image: {e2}")
                return Image.new('RGB', (512, 512), color='black')
    
    def split_latent_frequency_bands(self, latent: torch.Tensor, freq_config: FreqSepConfig, preserve_dc_component: bool = False, use_fft_shift: float = 1.0, use_correct_fft_shift: bool = False, mask_function: str = "center_circular", p=None) -> Dict[str, torch.Tensor]:
        """Split latent tensor into frequency bands using FFT"""
        try:
            # Convert BFloat16 to Float32 for FFT operations if needed
            original_dtype = latent.dtype
            if latent.dtype in (torch.bfloat16, torch.float16):
                latent = latent.float()
            
            # Apply FFT to latent channels
            latent_freq = torch.fft.fftn(latent, dim=(-2, -1))
            
            # Apply FFT shift if enabled to center low frequencies
            fft_shifted = use_fft_shift or use_correct_fft_shift
            if fft_shifted:
                latent_freq = torch.fft.fftshift(latent_freq, dim=(-2, -1))
                if use_fft_shift:
                    print(f"  🎯 Applied FFT shift (Detail Inspection mode - inverted masks)")
                if use_correct_fft_shift:
                    print(f"  🔧 Applied proper FFT shift with corrected masks")
            
            # Extract DC component (zero frequency) if preservation is enabled
            dc_component = None
            if preserve_dc_component:
                if fft_shifted:
                    # With fftshift, DC is in the center
                    h, w = latent_freq.shape[-2:]
                    dc_component = latent_freq[:, :, h//2, w//2].clone()
                else:
                    # Without fftshift, DC is at [0, 0]
                    dc_component = latent_freq[:, :, 0, 0].clone()
                print(f"  💡 Extracted DC component for preservation: {dc_component.shape}")
            
            frequency_bands = {}
            for band_config in freq_config.band_configs:
                # Create frequency mask for this band
                mask = self.create_latent_frequency_mask(
                    latent.shape[-2:], 
                    band_config.frequency_range, 
                    freq_config.overlap_factor,
                    mask_function
                )
                mask = mask.to(latent.device)
                
                # Debug: Save latent mask and spectrum
                if self.debug_mode:
                    print(f"  🔍 LATENT {band_config.name} mask stats: min={torch.min(mask):.3f}, max={torch.max(mask):.3f}, mean={torch.mean(mask):.3f}")
                    
                    # Save latent mask as image
                    mask_img = (mask.cpu().numpy() * 255).astype(np.uint8)
                    self._save_debug_image(mask_img, f"latent_{band_config.name}_mask.png", p, "LATENT")
                    
                    # Save latent spectrum magnitude for debugging
                    latent_spectrum_mag = torch.log(1 + torch.abs(latent_freq[0, 0]))  # First batch, first channel
                    latent_spectrum_mag = (latent_spectrum_mag / torch.max(latent_spectrum_mag) * 255).cpu().numpy().astype(np.uint8)
                    self._save_debug_image(latent_spectrum_mag, f"latent_{band_config.name}_input_spectrum.png", p, "LATENT")
                
                # Apply mask to get frequency band
                band_freq = latent_freq * mask.unsqueeze(0).unsqueeze(0)
                
                # Apply inverse FFT shift if needed before IFFT
                if use_fft_shift or use_correct_fft_shift:
                    # ALWAYS shift back to natural layout for proper IFFT reconstruction
                    band_freq = torch.fft.ifftshift(band_freq, dim=(-2, -1))
                
                band_latent = torch.fft.ifftn(band_freq, dim=(-2, -1)).real
                
                # Convert back to original dtype if needed
                if original_dtype != band_latent.dtype:
                    band_latent = band_latent.to(original_dtype)
                
                # Debug: Save masked spectrum and resulting band
                if self.debug_mode:
                    masked_spectrum_mag = torch.log(1 + torch.abs(band_freq[0, 0]))
                    masked_spectrum_mag = (masked_spectrum_mag / torch.max(masked_spectrum_mag) * 255).cpu().numpy().astype(np.uint8)
                    self._save_debug_image(masked_spectrum_mag, f"latent_{band_config.name}_masked_spectrum.png", p, "LATENT")
                
                frequency_bands[band_config.name] = band_latent
                
                # Debug: check if bands are actually different
                band_mean = torch.mean(band_latent).item()
                band_std = torch.std(band_latent).item()
                band_max = torch.max(band_latent).item()
                print(f"  🎵 {band_config.name}: {band_config.frequency_range} -> {band_latent.shape}")
                print(f"      📊 Stats: mean={band_mean:.3f}, std={band_std:.3f}, max={band_max:.3f}")
            
            # Store DC component for later use in recombination if preservation is enabled
            if preserve_dc_component and dc_component is not None:
                frequency_bands['_dc_component'] = dc_component
                print(f"  💡 Stored DC component for later preservation")
            
            return frequency_bands
            
        except Exception as e:
            print(f"⚠️ Error splitting latent frequency bands: {e}")
            # Fallback: return original latent for all bands
            fallback_bands = {}
            for band_config in freq_config.band_configs:
                fallback_bands[band_config.name] = latent.clone()
            return fallback_bands
    
    def create_latent_frequency_mask(self, shape: Tuple[int, int], freq_range: Tuple[float, float], 
                                   overlap_factor: float, mask_function: str = "center_circular") -> torch.Tensor:
        """Create frequency mask using various mathematical distance functions"""
        h, w = shape
        center_h, center_w = h // 2, w // 2
        
        # Create coordinate grids
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y, x = y.float(), x.float()
        
        # Calculate frequency magnitude based on chosen function
        if mask_function == "no_mask":
            # No frequency masking - all frequencies pass through
            print(f"    🚫 Using NO_MASK - all frequencies included")
            # Create a mask that's always 1.0 everywhere
            mask = torch.ones((h, w), dtype=torch.float32)
            return mask
        elif mask_function == "center_circular":
            # Simple circular distance from center (FFT shifted style)
            print(f"    🔧 Using CENTER CIRCULAR distance calculation")
            y_centered = y - center_h
            x_centered = x - center_w
            max_freq = math.sqrt(center_h**2 + center_w**2)
            freq_magnitude = torch.sqrt(x_centered**2 + y_centered**2) / max_freq
            
        elif mask_function == "corner_min_diamond":
            # Minimum distance to any corner (original no-shift style)
            print(f"    🎯 Using CORNER MIN (diamond) distance calculation")
            corners = [
                torch.sqrt(y**2 + x**2),  # distance from [0,0]
                torch.sqrt(y**2 + (x - w)**2),  # distance from [0,w]
                torch.sqrt((y - h)**2 + x**2),  # distance from [h,0]
                torch.sqrt((y - h)**2 + (x - w)**2)  # distance from [h,w]
            ]
            corner_dists = torch.stack(corners)
            freq_distances = torch.min(corner_dists, dim=0)[0]
            max_freq = math.sqrt(h**2 + w**2)
            freq_magnitude = freq_distances / max_freq
            
        elif mask_function == "corner_average":
            # Average distance to all corners (smooth alternative)
            print(f"    📊 Using CORNER AVERAGE distance calculation")
            corners = [
                torch.sqrt(y**2 + x**2),  # distance from [0,0]
                torch.sqrt(y**2 + (x - w)**2),  # distance from [0,w]
                torch.sqrt((y - h)**2 + x**2),  # distance from [h,0]
                torch.sqrt((y - h)**2 + (x - w)**2)  # distance from [h,w]
            ]
            freq_distances = torch.mean(torch.stack(corners), dim=0)
            max_freq = math.sqrt(h**2 + w**2)
            freq_magnitude = freq_distances / max_freq

        elif mask_function == "corner_rms":
            # RMS (Root Mean Square) distance to all corners
            print(f"    📐 Using CORNER RMS distance calculation")
            corners = [
                torch.sqrt(y**2 + x**2),  # distance from [0,0]
                torch.sqrt(y**2 + (x - w)**2),  # distance from [0,w]
                torch.sqrt((y - h)**2 + x**2),  # distance from [h,0]
                torch.sqrt((y - h)**2 + (x - w)**2)  # distance from [h,w]
            ]
            squared_distances = torch.stack([d**2 for d in corners])
            freq_distances = torch.sqrt(torch.mean(squared_distances, dim=0))
            max_freq = math.sqrt(h**2 + w**2)
            freq_magnitude = freq_distances / max_freq
            
        elif mask_function == "corner_harmonic":
            # Harmonic mean distance (emphasizes smaller distances)
            print(f"    🎵 Using CORNER HARMONIC distance calculation")
            corners = [
                torch.sqrt(y**2 + x**2),  # distance from [0,0]
                torch.sqrt(y**2 + (x - w)**2),  # distance from [0,w]
                torch.sqrt((y - h)**2 + x**2),  # distance from [h,0]
                torch.sqrt((y - h)**2 + (x - w)**2)  # distance from [h,w]
            ]
            # Harmonic mean: n / sum(1/x_i), with small epsilon to avoid division by zero
            epsilon = 1e-8
            reciprocals = torch.stack([1.0 / (d + epsilon) for d in corners])
            freq_distances = len(corners) / torch.sum(reciprocals, dim=0)
            max_freq = math.sqrt(h**2 + w**2)
            freq_magnitude = freq_distances / max_freq
            
        elif mask_function == "corner_product":
            # Product-based distance (creates interference patterns)
            print(f"    ✨ Using CORNER PRODUCT distance calculation")
            corners = [
                torch.sqrt(y**2 + x**2),  # distance from [0,0]
                torch.sqrt(y**2 + (x - w)**2),  # distance from [0,w]
                torch.sqrt((y - h)**2 + x**2),  # distance from [h,0]
                torch.sqrt((y - h)**2 + (x - w)**2)  # distance from [h,w]
            ]
            # Normalize each distance first, then take product
            normalized_corners = [d / torch.max(d) for d in corners]
            freq_distances = torch.prod(torch.stack(normalized_corners), dim=0)
            freq_magnitude = freq_distances  # Already normalized
            
        elif mask_function == "transformed_circular":
            # Circular mask with wraparound distance (equivalent to circular mask in FFT-shifted space)
            print(f"    🌀 Using TRANSFORMED CIRCULAR (wraparound) distance calculation")
            # Wrapped distance considering periodic boundaries
            dy = torch.minimum(torch.abs(y - center_h), h - torch.abs(y - center_h))
            dx = torch.minimum(torch.abs(x - center_w), w - torch.abs(x - center_w))
            
            # Euclidean distance with wraparound
            max_freq = math.sqrt((h/2)**2 + (w/2)**2)
            freq_magnitude = torch.sqrt(dx**2 + dy**2) / max_freq
            
        elif mask_function == "manhattan":
            # Manhattan distance (L1 norm) - creates diamond-shaped contours
            print(f"    💎 Using MANHATTAN distance calculation")
            y_centered = y - center_h
            x_centered = x - center_w
            max_freq = center_h + center_w  # Maximum possible L1 distance
            freq_magnitude = (torch.abs(x_centered) + torch.abs(y_centered)) / max_freq
            
        elif mask_function == "chebyshev":
            # Chebyshev distance (L∞ norm) - creates square-shaped contours
            print(f"    ⬜ Using CHEBYSHEV distance calculation")
            y_centered = y - center_h
            x_centered = x - center_w
            max_freq = max(center_h, center_w)  # Maximum possible L∞ distance
            freq_magnitude = torch.maximum(torch.abs(x_centered), torch.abs(y_centered)) / max_freq
            
        elif mask_function == "minkowski_1_5":
            # Minkowski distance with p=1.5 - interpolates between Manhattan and Euclidean
            print(f"    🔸 Using MINKOWSKI (p=1.5) distance calculation")
            y_centered = y - center_h
            x_centered = x - center_w
            p = 1.5
            # Minkowski: (|x|^p + |y|^p)^(1/p)
            freq_distances = torch.pow(torch.pow(torch.abs(x_centered), p) + torch.pow(torch.abs(y_centered), p), 1.0/p)
            max_freq = math.pow(center_h**p + center_w**p, 1.0/p)
            freq_magnitude = freq_distances / max_freq
            
        elif mask_function == "gravitational":
            # Gravitational potential - sum of 1/r from multiple mass points
            print(f"    🌌 Using GRAVITATIONAL potential calculation")
            # Place masses at corners and center
            mass_positions = [
                (0.0, 0.0),      # corner masses
                (0.0, float(w)),
                (float(h), 0.0),
                (float(h), float(w)),
                (center_h, center_w)  # center mass
            ]
            
            potential = torch.zeros_like(y, dtype=torch.float32)
            epsilon = 1e-6  # Avoid division by zero
            
            for mass_y, mass_x in mass_positions:
                r = torch.sqrt((y - mass_y)**2 + (x - mass_x)**2) + epsilon
                potential += 1.0 / r
            
            # Normalize by maximum potential
            max_potential = torch.max(potential)
            freq_magnitude = potential / max_potential
            
        elif mask_function == "wave_interference":
            # Wave interference pattern - sum of sin(k*r) from multiple sources
            print(f"    🌊 Using WAVE INTERFERENCE pattern calculation")
            # Wave sources at corners
            wave_sources = [
                (0.0, 0.0),
                (0.0, float(w)),
                (float(h), 0.0),
                (float(h), float(w))
            ]
            
            k = 2.0 * math.pi / min(h, w) * 4  # Wave number for ~4 wavelengths across image
            interference = torch.zeros_like(y, dtype=torch.float32)
            
            for source_y, source_x in wave_sources:
                r = torch.sqrt((y - source_y)**2 + (x - source_x)**2)
                interference += torch.sin(k * r)
            
            # Normalize to [0, 1] range
            interference = (interference - torch.min(interference)) / (torch.max(interference) - torch.min(interference))
            freq_magnitude = interference
            
        else:
            # Fallback to center circular
            print(f"    ⚠️ Unknown mask function '{mask_function}', using center_circular")
            y_centered = y - center_h
            x_centered = x - center_w
            max_freq = math.sqrt(center_h**2 + center_w**2)
            freq_magnitude = torch.sqrt(x_centered**2 + y_centered**2) / max_freq
        
        # universal 0-to-1 scaling
        freq_magnitude = freq_magnitude - freq_magnitude.min()        # shift so min = 0
        range_val = freq_magnitude.max()
        if range_val > 1e-8:                                          # avoid /0
            freq_magnitude /= range_val
        else:
            if hasattr(freq_magnitude, 'zero_'):  # torch
                freq_magnitude.zero_()
            else:  # numpy
                freq_magnitude.fill(0)

        low_freq, high_freq = freq_range
        
        # Create soft mask with smooth transitions
        mask = torch.zeros_like(freq_magnitude)
        transition_width = overlap_factor
        
        # Smooth transitions using sigmoid with very gentle slope
        sigmoid_sharpness = 1.0  # Very low for smooth, gradual transitions
        
        if low_freq == 0.0:
            # lowest band: keep everything inside
            lower_mask = torch.ones_like(freq_magnitude)
        else:
            lower_transition = (freq_magnitude - low_freq) / transition_width
            lower_mask = torch.sigmoid(lower_transition * sigmoid_sharpness)
        
        upper_transition = (high_freq - freq_magnitude) / transition_width  
        upper_mask = torch.sigmoid(upper_transition * sigmoid_sharpness)
        
        mask = lower_mask * upper_mask
        
        return mask
    
    def process_frequency_bands_in_latent_space(self, frequency_bands: Dict[str, torch.Tensor], 
                                              freq_config: FreqSepConfig, 
                                              p: StableDiffusionProcessingImg2Img, 
                                              image_index: int) -> Dict[str, torch.Tensor]:
        """Process each frequency band through ACTUAL diffusion with band-specific parameters"""
        
        from modules import shared
        
        processed_bands = {}
        low_freq_result = None  # For cross-attention sync
        shared_noise = None     # For synchronized noise
        
        # Generate shared noise for synchronization if needed
        if freq_config.sync_mode == SyncMode.SYNCHRONIZED_NOISE:
            # Create shared noise based on processing seed
            generator = torch.Generator(device=devices.device)
            if p.seed >= 0:
                generator.manual_seed(p.seed + image_index)
            else:
                generator.manual_seed(torch.randint(0, 2**32, (1,)).item())
            
            # Get noise shape from first band
            first_band = next(iter(frequency_bands.values()))
            shared_noise = torch.randn(first_band.shape, generator=generator, device=devices.device)
            print(f"  🎲 Generated synchronized noise: {shared_noise.shape}")
        
        # Filter out special keys like '_dc_component' before processing
        band_items = [(k, v) for k, v in frequency_bands.items() if not k.startswith('_')]
        
        # Sort bands for progressive processing if needed
        if freq_config.sync_mode == SyncMode.PROGRESSIVE:
            sorted_bands = sorted(band_items, 
                                key=lambda x: next(cfg.frequency_range[0] for cfg in freq_config.band_configs if cfg.name == x[0]))
        else:
            sorted_bands = band_items
        
        # Update job count for progress tracking
        total_bands = len(sorted_bands)
        original_job_count = shared.state.job_count
        shared.state.job_count += total_bands
        print(f"  📈 Added {total_bands} frequency band diffusion jobs to queue")
        
        for band_name, band_latent in sorted_bands:
            band_config = next(cfg for cfg in freq_config.band_configs if cfg.name == band_name)
            
            # Check for interruption
            if shared.state.interrupted or shared.state.skipped:
                print(f"  ⏹️ Processing interrupted, stopping at {band_name} band")
                break
            
            print(f"  🎛️ REAL DIFFUSION processing {band_name} band (denoising: {band_config.denoising_strength}, steps: {band_config.steps}, cfg: {band_config.cfg_scale})")
            print(f"     📊 Band config: steps={band_config.steps}, cfg={band_config.cfg_scale}")
            
            try:
                # Run ACTUAL diffusion with progress bars!
                processed_latent = self.run_diffusion_on_frequency_band(
                    band_latent, band_config, p, freq_config.sync_mode, 
                    shared_noise, low_freq_result, image_index
                )
                
                processed_bands[band_name] = processed_latent
                print(f"    ✅ {band_name} band REAL diffusion complete")
                
                # Store low frequency result for cross-attention guidance
                if band_config.preserve_composition and freq_config.sync_mode == SyncMode.CROSS_ATTENTION:
                    low_freq_result = processed_latent
                    print(f"    🎯 Stored {band_name} as guidance for higher frequencies")
                
                # Update progress
                shared.state.current_image_sampling_step += 1
                
            except Exception as e:
                print(f"    ⚠️ Error in REAL diffusion processing {band_name} band: {e}, using enhanced fallback")
                import traceback
                traceback.print_exc()
                
                # Fallback to enhanced latent
                processed_bands[band_name] = self.apply_latent_enhancement(
                    band_latent, band_config, freq_config.sync_mode
                )
        
        print(f"  🎆 All frequency bands processed! {len(processed_bands)}/{total_bands} successful")
        
        # Restore DC component if it was preserved
        if '_dc_component' in frequency_bands:
            dc_component = frequency_bands['_dc_component']
            print(f"  💡 Restoring preserved DC component to processed bands...")
            
            for band_name, band_latent in processed_bands.items():
                # Convert back to frequency domain to restore DC component
                band_freq = torch.fft.fftn(band_latent, dim=(-2, -1))
                
                # Restore the original DC component
                # band_freq[:, :, 0, 0] = dc_component
                
                h, w = band_freq.shape[-2:]
                band_freq[:, :, h//2, w//2] = dc_component   # centre pixel

                # Convert back to latent space
                restored_band = torch.fft.ifftn(band_freq, dim=(-2, -1)).real
                processed_bands[band_name] = restored_band
                
                print(f"    ✅ Restored DC component to {band_name} band")
        
        return processed_bands
    
    def run_diffusion_on_frequency_band(self, band_latent: torch.Tensor, 
                                       band_config: FrequencyBandConfig,
                                       base_p: StableDiffusionProcessingImg2Img,
                                       sync_mode: SyncMode,
                                       shared_noise: Optional[torch.Tensor],
                                       guidance_latent: Optional[torch.Tensor],
                                       image_index: int) -> torch.Tensor:
        """Run ACTUAL diffusion denoising using WebUI's built-in processing (like ADetailer)"""
        
        try:
            from modules import processing
            from modules.processing import StableDiffusionProcessingImg2Img
            from copy import copy
            
            print(f"      🔥 Starting REAL diffusion on {band_config.name} band...")
            
            # Decode the frequency band latent to an image for img2img processing
            band_image = self.decode_latent_to_image(band_latent, base_p)
            print(f"      🖼️ Decoded {band_config.name} band to image: {band_image.size}")
            
            # Apply cross-attention guidance in image space if available
            if guidance_latent is not None and not band_config.preserve_composition:
                guidance_image = self.decode_latent_to_image(guidance_latent, base_p)
                guidance_strength = 0.3
                
                # Blend images in PIL space
                import numpy as np
                from PIL import Image
                
                band_array = np.array(band_image).astype(np.float32)
                guidance_array = np.array(guidance_image).astype(np.float32)
                
                blended_array = (1 - guidance_strength) * band_array + guidance_strength * guidance_array
                blended_array = np.clip(blended_array, 0, 255).astype(np.uint8)
                band_image = Image.fromarray(blended_array)
                
                print(f"      🎯 Applied cross-attention guidance (strength: {guidance_strength})")
            
            # Create img2img processing object following ADetailer's approach
            band_seed = base_p.seed + hash(band_config.name) if base_p.seed >= 0 else base_p.seed
            if sync_mode == SyncMode.SYNCHRONIZED_NOISE:
                band_seed = base_p.seed  # Use same seed for synchronization
            
            # Copy extra generation params from base processing
            extra_params = {}
            if hasattr(base_p, 'extra_generation_params'):
                extra_params = copy(base_p.extra_generation_params)
            
            # Create the img2img processing object (following ADetailer pattern)
            band_p = StableDiffusionProcessingImg2Img(
                init_images=[band_image],
                resize_mode=0,
                denoising_strength=band_config.denoising_strength,
                mask=None,
                mask_blur=0,
                inpainting_fill=1,
                inpaint_full_res=False,
                inpaint_full_res_padding=0,
                inpainting_mask_invert=0,
                initial_noise_multiplier=None,
                sd_model=base_p.sd_model,
                outpath_samples=getattr(base_p, 'outpath_samples', 'outputs/img2img-images'),
                outpath_grids=getattr(base_p, 'outpath_grids', 'outputs/img2img-grids'),
                prompt=base_p.prompt,
                negative_prompt=base_p.negative_prompt,
                styles=getattr(base_p, 'styles', []),
                seed=band_seed,
                subseed=getattr(base_p, 'subseed', -1),
                subseed_strength=getattr(base_p, 'subseed_strength', 0),
                seed_resize_from_h=getattr(base_p, 'seed_resize_from_h', -1),
                seed_resize_from_w=getattr(base_p, 'seed_resize_from_w', -1),
                sampler_name=base_p.sampler_name,
                batch_size=1,
                n_iter=1,
                steps=band_config.steps,
                cfg_scale=band_config.cfg_scale,
                width=getattr(base_p, 'width', 512),
                height=getattr(base_p, 'height', 512),
                restore_faces=getattr(base_p, 'restore_faces', False),
                tiling=getattr(base_p, 'tiling', False),
                extra_generation_params=extra_params,
                do_not_save_samples=True,
                do_not_save_grid=True,
                override_settings={}
            )
            
            # Prevent preview JPGs from being written during band processing
            band_p.save_images = False      # For WebUI >= 1.6
            if hasattr(band_p, 'save_preview'):
                band_p.save_preview = False  # For older builds
            
            # Important: Clear cached conditioning (like ADetailer does)
            band_p.cached_c = [None, None]
            band_p.cached_uc = [None, None]
            
            # Add missing hr attributes to prevent infotext creation errors
            # These are needed for create_infotext() to work properly
            if not hasattr(band_p, 'all_hr_prompts'):
                band_p.all_hr_prompts = [band_p.prompt]
            if not hasattr(band_p, 'all_hr_negative_prompts'):
                band_p.all_hr_negative_prompts = [band_p.negative_prompt]
            
            # Copy scripts from base processing (essential for proper operation)
            if hasattr(base_p, 'scripts') and base_p.scripts is not None:
                from copy import copy
                band_p.scripts = copy(base_p.scripts)
                band_p.script_args = copy(base_p.script_args) if hasattr(base_p, 'script_args') else []
                print(f"      🔧 Copied scripts from base processing: {len(band_p.script_args) if band_p.script_args else 0} script args")
            else:
                print(f"      ⚠️ No scripts found on base processing object")
                # Create minimal scripts object to prevent the error
                try:
                    from modules import scripts
                    band_p.scripts = scripts.ScriptRunner()
                    band_p.script_args = []
                except Exception as e:
                    print(f"      ⚠️ Could not create minimal scripts: {e}")
            
            # Disable this extension for the band processing to avoid recursion
            band_p._ad_disabled = True
            band_p._frequency_separation_disabled = True
            
            print(f"      ⚙️ Running diffusion: {band_config.steps} steps, denoising {band_config.denoising_strength:.2f}, CFG {band_config.cfg_scale}, seed {band_seed}")
            print(f"      🔧 Processing object ready: scripts={band_p.scripts is not None}, script_args={len(band_p.script_args) if hasattr(band_p, 'script_args') else 'None'}")
            
            # Create frequency mask for this band in latent space
            latent_height, latent_width = band_latent.shape[-2:]
            freq_mask = self.create_latent_frequency_mask(
                (latent_height, latent_width),
                band_config.frequency_range,
                base_p._freq_sep_config.overlap_factor if hasattr(base_p, '_freq_sep_config') else 0.1,
                base_p._freq_sep_mask_function if hasattr(base_p, '_freq_sep_mask_function') else 'center_circular'
            )
            
            # Enable frequency clamping for this band
            frequency_clamping_callback.set_frequency_constraint(
                freq_mask, 
                band_config.name,
                use_fft_shift=getattr(base_p, '_freq_sep_use_fft_shift', False)
            )
            print(f"      🎯 Enabled frequency clamping for {band_config.name} band")
            
            # Run the ACTUAL diffusion process with progress bars (ADetailer approach)!
            try:
                print(f"      🚀 Starting processing.process_images() for {band_config.name}...")
                processed = processing.process_images(band_p)
                print(f"      ✅ process_images() completed for {band_config.name}")
            except Exception as processing_error:
                print(f"      ⚠️ Processing failed: {processing_error}")
                import traceback
                traceback.print_exc()
                return band_latent
            finally:
                # Disable frequency clamping
                frequency_clamping_callback.disable()
                print(f"      🎯 Disabled frequency clamping")
                
                # Clean up (like ADetailer does)
                if hasattr(band_p, 'close'):
                    band_p.close()
            
            if not processed or not processed.images:
                print(f"      ⚠️ Diffusion processing failed for {band_config.name} - no images returned")
                return band_latent
            
            # Get the processed image and encode it back to latent
            processed_image = processed.images[0]
            processed_latent = self.encode_image_to_latent(processed_image, base_p)
            
            print(f"      ✅ {band_config.name} diffusion complete! Shape: {processed_latent.shape}")
            return processed_latent
            
        except Exception as e:
            print(f"      ⚠️ Real diffusion failed for {band_config.name}: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to enhanced version
            return self.apply_latent_enhancement(band_latent, band_config, sync_mode)
    
    def simulate_diffusion_denoising(self, noised_latent: torch.Tensor, 
                                   original_latent: torch.Tensor,
                                   band_config: FrequencyBandConfig,
                                   base_p: StableDiffusionProcessingImg2Img) -> torch.Tensor:
        """DEPRECATED: Use run_diffusion_on_frequency_band for actual diffusion"""
        
        # This method is now deprecated in favor of actual diffusion processing
        # Only used as fallback if real diffusion fails
        
        print(f"        ⚠️ Using fallback simulation for {band_config.name}")
        
        denoising_strength = band_config.denoising_strength
        
        if band_config.preserve_composition:
            # Low frequency: gentle enhancement to preserve structure
            enhancement_factor = 1.0 + (denoising_strength * 0.1)
            denoised = original_latent * enhancement_factor
            print(f"        🏗️ Structure enhancement: factor={enhancement_factor:.2f}")
        else:
            # Mid/High frequency: stronger enhancement for details
            enhancement_factor = 1.0 + (denoising_strength * 0.3)
            denoised = original_latent * enhancement_factor
            
            # Add controlled noise for texture enhancement
            noise_strength = denoising_strength * 0.05
            texture_noise = torch.randn_like(original_latent) * noise_strength
            denoised = denoised + texture_noise
            
            print(f"        ✨ Detail enhancement: factor={enhancement_factor:.2f}, noise={noise_strength:.3f}")
        
        return denoised
    
    def apply_latent_enhancement(self, latent: torch.Tensor, band_config: FrequencyBandConfig, 
                               sync_mode: SyncMode) -> torch.Tensor:
        """Apply enhancement to latent based on band configuration"""
        try:
            enhanced_latent = latent.clone()
            
            print(f"      🔧 Enhancing {band_config.name} with denoising={band_config.denoising_strength}")
            
            if band_config.preserve_composition:
                # Low frequency: preserve structure, slight contrast enhancement
                enhancement_factor = 1.0 + (band_config.denoising_strength * 0.2)
                enhanced_latent = enhanced_latent * enhancement_factor
                print(f"        📐 Low freq enhancement factor: {enhancement_factor:.2f}")
            else:
                # High frequency: stronger enhancement for details
                enhancement_factor = 1.0 + (band_config.denoising_strength * 0.5)
                enhanced_latent = enhanced_latent * enhancement_factor
                
                # Add controlled noise for texture enhancement
                noise_strength = band_config.denoising_strength * 0.1
                noise = torch.randn_like(enhanced_latent) * noise_strength
                enhanced_latent = enhanced_latent + noise
                
                # Apply sharpening effect for high frequencies
                if band_config.name == "high_freq":
                    # Additional sharpening for high frequency bands
                    sharpening_factor = 1.0 + (band_config.denoising_strength * 0.3)
                    enhanced_latent = enhanced_latent * sharpening_factor
                    print(f"        ✨ High freq sharpening: {sharpening_factor:.2f}, noise: {noise_strength:.3f}")
                else:
                    print(f"        🎯 Mid freq enhancement: {enhancement_factor:.2f}, noise: {noise_strength:.3f}")
            
            return enhanced_latent
            
        except Exception as e:
            print(f"⚠️ Error in latent enhancement: {e}")
            return latent
    
    def recombine_decoded_frequency_bands(self, decoded_bands: Dict[str, Image.Image], 
                                        freq_config: FreqSepConfig, recombination_method: str, mask_function: str = "center_circular") -> Image.Image:
        """
        Recombine frequency bands using either frequency reconstruction or simple blending.
        This defeats VAE limitations by combining details from multiple VAE interpretations!
        """
        try:
            print(f"    🎨 Starting multi-VAE image space recombination using {recombination_method}...")
            
            # Convert all PIL images to numpy arrays and check differences
            band_arrays = {}
            for band_name, image in decoded_bands.items():
                band_arrays[band_name] = np.array(image).astype(np.float32)
                avg_intensity = np.mean(band_arrays[band_name])
                std_intensity = np.std(band_arrays[band_name])
                print(f"      📐 {band_name}: {band_arrays[band_name].shape} avg={avg_intensity:.1f} std={std_intensity:.1f}")
            
            # Get image dimensions from first band
            first_band = next(iter(band_arrays.values()))
            height, width, channels = first_band.shape
            
            if recombination_method == "frequency_reconstruction":
                print("    🌊 Using FREQUENCY RECONSTRUCTION with energy normalization...")
                final_image = self._frequency_reconstruction_recombination(band_arrays, freq_config, height, width, channels, mask_function)
            else:  # simple_blending
                print("    🌊 Using SIMPLE weighted blending...")
                final_image = self._simple_blending_recombination(band_arrays, freq_config, height, width, channels)
            
            print(f"    🎉 Multi-VAE recombination successful: {final_image.size}")
            return final_image
            
        except Exception as e:
            print(f"⚠️ Error in multi-VAE recombination: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback: simple average of decoded images
            print("    🔄 Falling back to simple image averaging...")
            try:
                images_array = [np.array(img).astype(np.float32) for img in decoded_bands.values()]
                averaged = np.mean(images_array, axis=0).astype(np.uint8)
                return Image.fromarray(averaged, 'RGB')
            except Exception as e2:
                print(f"⚠️ Fallback failed: {e2}")
                # Ultimate fallback: return first image
                return next(iter(decoded_bands.values()))
    
    def create_image_frequency_mask(self, shape: Tuple[int, int], freq_range: Tuple[float, float], 
                                  overlap_factor: float, mask_function: str = "center_circular") -> np.ndarray:
        """Create frequency mask for image space using various mathematical distance functions"""
        h, w = shape
        center_h, center_w = h // 2, w // 2
        
        # Create coordinate grids
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        y, x = y.astype(np.float32), x.astype(np.float32)
        
        # Calculate frequency magnitude based on chosen function
        if mask_function == "no_mask":
            # No frequency masking - all frequencies pass through
            print(f"    🚫 Using NO_MASK - all frequencies included (image space)")
            # Create a mask that's always 1.0 everywhere
            mask = np.ones((h, w), dtype=np.float32)
            return mask
        elif mask_function == "center_circular":
            # Simple circular distance from center (FFT shifted style)
            print(f"    🔧 Using CENTER CIRCULAR distance calculation (image space)")
            y_centered = y - center_h
            x_centered = x - center_w
            max_freq = np.sqrt(center_h**2 + center_w**2)
            freq_magnitude = np.sqrt(x_centered**2 + y_centered**2) / max_freq
            
        elif mask_function == "corner_min_diamond":
            # Minimum distance to any corner (original no-shift style)
            print(f"    🎯 Using CORNER MIN (diamond) distance calculation (image space)")
            corners = [
                np.sqrt(y**2 + x**2),  # distance from [0,0]
                np.sqrt(y**2 + (x - w)**2),  # distance from [0,w]
                np.sqrt((y - h)**2 + x**2),  # distance from [h,0]
                np.sqrt((y - h)**2 + (x - w)**2)  # distance from [h,w]
            ]
            corner_dists = np.stack(corners)
            freq_distances = np.min(corner_dists, axis=0)
            max_freq = np.sqrt(h**2 + w**2)
            freq_magnitude = freq_distances / max_freq
            
        elif mask_function == "corner_average":
            # Average distance to all corners (smooth alternative)
            print(f"    📊 Using CORNER AVERAGE distance calculation (image space)")
            corners = [
                np.sqrt(y**2 + x**2),  # distance from [0,0]
                np.sqrt(y**2 + (x - w)**2),  # distance from [0,w]
                np.sqrt((y - h)**2 + x**2),  # distance from [h,0]
                np.sqrt((y - h)**2 + (x - w)**2)  # distance from [h,w]
            ]
            freq_distances = np.mean(np.stack(corners), axis=0)
            max_freq = np.sqrt(h**2 + w**2)
            freq_magnitude = freq_distances / max_freq
            
        elif mask_function == "corner_rms":
            # RMS (Root Mean Square) distance to all corners
            print(f"    📐 Using CORNER RMS distance calculation (image space)")
            corners = [
                np.sqrt(y**2 + x**2),  # distance from [0,0]
                np.sqrt(y**2 + (x - w)**2),  # distance from [0,w]
                np.sqrt((y - h)**2 + x**2),  # distance from [h,0]
                np.sqrt((y - h)**2 + (x - w)**2)  # distance from [h,w]
            ]
            squared_distances = np.stack([d**2 for d in corners])
            freq_distances = np.sqrt(np.mean(squared_distances, axis=0))
            max_freq = np.sqrt(h**2 + w**2)
            freq_magnitude = freq_distances / max_freq
            
        elif mask_function == "corner_harmonic":
            # Harmonic mean distance (emphasizes smaller distances)
            print(f"    🎵 Using CORNER HARMONIC distance calculation (image space)")
            epsilon = 1e-8
            corners = [
                np.sqrt(y**2 + x**2),  # distance from [0,0]
                np.sqrt(y**2 + (x - w)**2),  # distance from [0,w]
                np.sqrt((y - h)**2 + x**2),  # distance from [h,0]
                np.sqrt((y - h)**2 + (x - w)**2)  # distance from [h,w]
            ]
            reciprocals = np.stack([1.0 / (d + epsilon) for d in corners])
            freq_distances = len(corners) / np.sum(reciprocals, axis=0)
            max_freq = np.sqrt(h**2 + w**2)
            freq_magnitude = freq_distances / max_freq
            
        elif mask_function == "corner_product":
            # Product-based distance (creates interference patterns)
            print(f"    ✨ Using CORNER PRODUCT distance calculation (image space)")
            corners = [
                np.sqrt(y**2 + x**2),  # distance from [0,0]
                np.sqrt(y**2 + (x - w)**2),  # distance from [0,w]
                np.sqrt((y - h)**2 + x**2),  # distance from [h,0]
                np.sqrt((y - h)**2 + (x - w)**2)  # distance from [h,w]
            ]
            # Normalize each distance first, then take product
            normalized_corners = [d / np.max(d) for d in corners]
            freq_distances = np.prod(np.stack(normalized_corners), axis=0)
            freq_magnitude = freq_distances  # Already normalized
            
        elif mask_function == "transformed_circular":
            # Circular mask with wraparound distance (equivalent to circular mask in FFT-shifted space)
            print(f"    🌀 Using TRANSFORMED CIRCULAR (wraparound) distance calculation (image space)")
            # Wrapped distance considering periodic boundaries
            dy = np.minimum(np.abs(y - center_h), h - np.abs(y - center_h))
            dx = np.minimum(np.abs(x - center_w), w - np.abs(x - center_w))
            
            # Euclidean distance with wraparound
            max_freq = np.sqrt((h//2)**2 + (w//2)**2)
            freq_magnitude = np.sqrt(dx**2 + dy**2) / max_freq
            
        elif mask_function == "manhattan":
            # Manhattan distance (L1 norm) - creates diamond-shaped contours
            print(f"    💎 Using MANHATTAN distance calculation (image space)")
            y_centered = y - center_h
            x_centered = x - center_w
            max_freq = center_h + center_w  # Maximum possible L1 distance
            freq_magnitude = (np.abs(x_centered) + np.abs(y_centered)) / max_freq
            
        elif mask_function == "chebyshev":
            # Chebyshev distance (L∞ norm) - creates square-shaped contours
            print(f"    ⬜ Using CHEBYSHEV distance calculation (image space)")
            y_centered = y - center_h
            x_centered = x - center_w
            max_freq = max(center_h, center_w)  # Maximum possible L∞ distance
            freq_magnitude = np.maximum(np.abs(x_centered), np.abs(y_centered)) / max_freq
            
        elif mask_function == "minkowski_1_5":
            # Minkowski distance with p=1.5 - interpolates between Manhattan and Euclidean
            print(f"    🔸 Using MINKOWSKI (p=1.5) distance calculation (image space)")
            y_centered = y - center_h
            x_centered = x - center_w
            p = 1.5
            # Minkowski: (|x|^p + |y|^p)^(1/p)
            freq_distances = np.power(np.power(np.abs(x_centered), p) + np.power(np.abs(y_centered), p), 1.0/p)
            max_freq = np.power(center_h**p + center_w**p, 1.0/p)
            freq_magnitude = freq_distances / max_freq
            
        elif mask_function == "gravitational":
            # Gravitational potential - sum of 1/r from multiple mass points
            print(f"    🌌 Using GRAVITATIONAL potential calculation (image space)")
            # Place masses at corners and center
            mass_positions = [
                (0.0, 0.0),      # corner masses
                (0.0, float(w)), 
                (float(h), 0.0), 
                (float(h), float(w)),
                (center_h, center_w)  # center mass
            ]
            
            epsilon = 1e-8
            potential = np.zeros_like(y, dtype=np.float32)
            
            for mass_y, mass_x in mass_positions:
                distance = np.sqrt((y - mass_y)**2 + (x - mass_x)**2) + epsilon
                potential += 1.0 / distance
            
            # Normalize by maximum potential
            max_potential = np.max(potential)
            freq_magnitude = potential / max_potential
            
        elif mask_function == "wave_interference":
            # Wave interference pattern - sum of sin(k*r) from multiple sources
            print(f"    🌊 Using WAVE INTERFERENCE pattern calculation (image space)")
            # Wave sources at corners
            wave_sources = [
                (0.0, 0.0),
                (0.0, float(w)),
                (float(h), 0.0), 
                (float(h), float(w))
            ]
            
            k = 2.0 * np.pi / min(h, w) * 4  # Wave number
            interference = np.zeros_like(y, dtype=np.float32)
            
            for source_y, source_x in wave_sources:
                distance = np.sqrt((y - source_y)**2 + (x - source_x)**2)
                wave = np.sin(k * distance)
                interference += wave
            
            # Normalize to [0, 1] range
            interference_normalized = (interference - np.min(interference)) / (np.max(interference) - np.min(interference))
            freq_magnitude = interference_normalized
            
        else:
            # Fallback to center_circular
            print(f"    ⚠️ Unknown mask function '{mask_function}', falling back to center_circular (image space)")
            y_centered = y - center_h
            x_centered = x - center_w
            max_freq = np.sqrt(center_h**2 + center_w**2)
            freq_magnitude = np.sqrt(x_centered**2 + y_centered**2) / max_freq
        
        # --- make magnitude span 0-1, whatever the distance function -------------
        freq_magnitude = freq_magnitude - freq_magnitude.min()
        range_val = freq_magnitude.max()
        if range_val > 1e-8:
            freq_magnitude /= range_val
        else:                                   # completely flat – shouldn't happen
            freq_magnitude.fill(0.0)
        # -------------------------------------------------------------------------
        
        low_freq, high_freq = freq_range
        
        # Create soft mask with smooth transitions
        transition_width = overlap_factor
        
        # Smooth transitions using sigmoid-like function with very gentle slope
        sigmoid_sharpness = 1.0  # Very low for smooth, gradual transitions
        
        if low_freq == 0.0:
            # lowest band: keep everything inside
            lower_mask = np.ones_like(freq_magnitude)
        else:
            lower_transition = (freq_magnitude - low_freq) / transition_width
            lower_mask = 1.0 / (1.0 + np.exp(-lower_transition * sigmoid_sharpness))
        
        upper_transition = (high_freq - freq_magnitude) / transition_width  
        upper_mask = 1.0 / (1.0 + np.exp(-upper_transition * sigmoid_sharpness))
        
        mask = lower_mask * upper_mask
        
        return mask
    
    def _frequency_reconstruction_recombination(self, band_arrays: Dict[str, np.ndarray], 
                                              freq_config: FreqSepConfig, height: int, width: int, channels: int, mask_function: str = "center_circular") -> Image.Image:
        """Recombine frequency bands using FFT reconstruction with energy normalization"""
        try:
            # # Store original energy for normalization
            # original_energies = {}
            # for band_name, band_array in band_arrays.items():
            #     original_energies[band_name] = np.mean(band_array ** 2)  # Energy = mean squared intensity
            # total_original_energy = sum(original_energies.values())
            
            # print(f"      🔋 Original energy per band: {', '.join([f'{name}={energy:.1f}' for name, energy in original_energies.items()])}")
            # print(f"      🔋 Total original energy: {total_original_energy:.1f}")
            
            # Initialize combined frequency domain representation
            combined_freq = None          # complex array  (H,W,C)
            sum_mask      = None          # real array     (H,W)
            
            # First pass: create all masks and calculate their sum
            all_masks = {}
            for band_name in band_arrays.keys():
                band_config = next(cfg for cfg in freq_config.band_configs if cfg.name == band_name)
                mask = self.create_image_frequency_mask(
                    (height, width),
                    band_config.frequency_range,
                    freq_config.overlap_factor,
                    mask_function
                )
                amplitude_scale = band_config.amplitude_scale
                all_masks[band_name] = mask * amplitude_scale
            
            # Calculate the sum of all masks
            total_mask = np.zeros((height, width))
            for mask in all_masks.values():
                total_mask += mask
            
            # Normalize masks so they sum to 1.0
            print(f"      📊 Pre-normalization mask sum: min={np.min(total_mask):.3f}, max={np.max(total_mask):.3f}, mean={np.mean(total_mask):.3f}")
            for band_name in all_masks:
                all_masks[band_name] = np.where(total_mask > 0, all_masks[band_name] / total_mask, 0)
            
            # Verify normalization
            normalized_sum = np.zeros((height, width))
            for mask in all_masks.values():
                normalized_sum += mask
            print(f"      ✅ Post-normalization mask sum: min={np.min(normalized_sum):.3f}, max={np.max(normalized_sum):.3f}, mean={np.mean(normalized_sum):.3f}")
            
            for band_name, band_array in band_arrays.items():
                band_config = next(cfg for cfg in freq_config.band_configs if cfg.name == band_name)
                
                print(f"      🎵 Processing {band_name} for frequency reconstruction...")
                
                # Convert each channel to frequency domain
                band_freq_channels = []
                for c in range(channels):
                    # Apply FFT to this channel
                    channel_freq = np.fft.fft2(band_array[:, :, c])
                    band_freq_channels.append(channel_freq)

                # -- build complex spectrum for this band ----------------------------
                band_freq = np.stack([np.fft.fft2(band_array[:, :, c])
                                    for c in range(channels)], axis=-1)

                # Use pre-normalized mask
                mask = all_masks[band_name]
                
                # Debug: Save mask and spectrum images
                if self.debug_mode:
                    print(f"      🔍 {band_name} normalized mask stats: min={np.min(mask):.3f}, max={np.max(mask):.3f}, mean={np.mean(mask):.3f}")
                    
                    # Save mask as image for debugging
                    mask_img = (mask * 255).astype(np.uint8)
                    self._save_debug_image(mask_img, f"image_{band_name}_mask.png", None, "IMAGE")
                    
                    # Save spectrum magnitude for debugging
                    spectrum_mag = np.log(1 + np.abs(band_freq[:,:,0]))  # Log scale for visibility
                    spectrum_mag = (spectrum_mag / np.max(spectrum_mag) * 255).astype(np.uint8)
                    self._save_debug_image(spectrum_mag, f"image_{band_name}_spectrum.png", None, "IMAGE")

                # Use the normalized mask directly (amplitude scale already applied)
                masked_scaled_freq = band_freq * mask[..., None]
                
                if combined_freq is None:
                    combined_freq = masked_scaled_freq
                    sum_mask      = mask
                else:
                    combined_freq += masked_scaled_freq
                    sum_mask      += mask
                
                # Debug: Save masked spectrum to see what's actually being accumulated
                if self.debug_mode:
                    # Show the actual masked and scaled spectrum that's being used
                    masked_mag = np.log1p(np.abs(masked_scaled_freq[:,:,0]))
                    if masked_mag.max() > 0:  # Avoid division by zero
                        masked_mag = (masked_mag / masked_mag.max() * 255).astype(np.uint8)
                    else:
                        masked_mag = np.zeros_like(masked_mag, dtype=np.uint8)
                    self._save_debug_image(masked_mag,
                                          f"image_{band_name}_masked_spectrum.png",
                                          None, "IMAGE")
            
            # Debug: Check sum_mask statistics (should be 1.0 everywhere now)
            print(f"      📊 Final sum_mask stats: min={np.min(sum_mask):.3f}, max={np.max(sum_mask):.3f}, mean={np.mean(sum_mask):.3f}")
            
            # Check how many pixels have sum_mask != 1.0
            deviation_from_one = np.abs(sum_mask - 1.0)
            pixels_not_one = np.sum(deviation_from_one > 0.01)
            total_pixels = sum_mask.size
            print(f"      ✅ Pixels where sum_mask deviates from 1.0 by >0.01: {pixels_not_one}/{total_pixels} ({pixels_not_one/total_pixels*100:.1f}%)")
            
            # No need to divide by sum_mask since masks are pre-normalized to sum to 1.0
            # combined_freq already contains the properly weighted sum

            # Convert back to spatial domain
            final_channels = []
            for c in range(channels):
                # Apply inverse FFT
                channel_spatial = np.fft.ifft2(combined_freq[:, :, c]).real
                final_channels.append(channel_spatial)
            
            # Stack channels back together
            reconstructed_image = np.stack(final_channels, axis=-1)
            
            # ENERGY NORMALIZATION: DISABLED - Now handled in latent space before VAE decoding
            # This avoids double energy correction which was causing over-brightening
            # reconstructed_energy = np.mean(reconstructed_image ** 2)
            # energy_ratio = total_original_energy / reconstructed_energy if reconstructed_energy > 0 else 1.0
            
            # print(f"      🔋 Reconstructed energy: {reconstructed_energy:.1f} (info only)")
            # print(f"      ⚡ Energy ratio (original/reconstructed): {energy_ratio:.3f} (info only)")
            # print(f"      📝 Image-space energy normalization DISABLED - using latent-space normalization instead")
            
            # Debug: Show final brightness for comparison (without energy correction)
            final_brightness_check = np.mean(reconstructed_image)
            original_brightness = np.mean([np.mean(arr) for arr in band_arrays.values()])
            print(f"      💡 Final brightness (no energy correction): {final_brightness_check:.1f} (original: {original_brightness:.1f})")
            
            # Clamp and convert back to uint8
            reconstructed_image = np.clip(reconstructed_image, 0, 255).astype(np.uint8)
            
            # Convert back to PIL
            final_image = Image.fromarray(reconstructed_image, 'RGB')
            
            # Final energy check
            final_brightness = np.mean(np.array(final_image))
            print(f"      🎯 Final brightness: {final_brightness:.1f} (target: {original_brightness:.1f})")
            
            return final_image
            
        except Exception as e:
            print(f"      ⚠️ Frequency reconstruction failed: {e}, falling back to simple blending")
            return self._simple_blending_recombination(band_arrays, freq_config, height, width, channels)
    
    def _simple_blending_recombination(self, band_arrays: Dict[str, np.ndarray], 
                                     freq_config: FreqSepConfig, height: int, width: int, channels: int) -> Image.Image:
        """Recombine frequency bands using simple weighted blending"""
        combined_image = np.zeros_like(next(iter(band_arrays.values())))
        total_weight = 0.0
        
        for band_name, band_array in band_arrays.items():
            band_config = next(cfg for cfg in freq_config.band_configs if cfg.name == band_name)
            
            # Simple weighting based on band characteristics
            if band_config.preserve_composition:
                weight = 0.5  # Structure contribution
                print(f"      🏗️ {band_name} (structure): weight {weight:.1f}")
            elif band_config.name == "high_freq":
                weight = 0.4  # Detail contribution
                print(f"      ✨ {band_name} (details): weight {weight:.1f}")
            else:
                weight = 0.3  # Feature contribution
                print(f"      🎯 {band_name} (features): weight {weight:.1f}")
            
            # Apply amplitude scaling
            amplitude_scale = band_config.amplitude_scale
            print(f"        🎚️ Applying amplitude scale {amplitude_scale:.2f} to {band_name}")
            
            # Apply both weight and amplitude scaling
            effective_weight = weight * amplitude_scale
            combined_image += band_array * effective_weight
            total_weight += effective_weight
        
        # Debug: Check brightness before and after normalization
        pre_norm_brightness = np.mean(combined_image)
        print(f"      💡 Before normalization: brightness={pre_norm_brightness:.1f}, total_weight={total_weight:.2f}")
        
        # Normalize by total weight
        combined_image /= total_weight
        
        post_norm_brightness = np.mean(combined_image)
        print(f"      💡 After normalization: brightness={post_norm_brightness:.1f}")
        print(f"      🔍 Normalization effect: {pre_norm_brightness:.1f} → {post_norm_brightness:.1f} (ratio: {post_norm_brightness/pre_norm_brightness:.3f})")
        
        # Clamp and convert back to uint8
        combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)
        
        # Convert back to PIL
        final_image = Image.fromarray(combined_image, 'RGB')
        
        return final_image
    
    def postprocess_batch(self, p, *args, **kwargs):
        """Process the entire batch before ADetailer gets individual images"""
        
        # Check if this is a txt2img run with frequency separation enabled
        if not getattr(p, '_freq_sep_enabled', False):
            return
            
        # Prevent recursion from temporary processing objects created during frequency band processing
        if getattr(p, '_frequency_separation_disabled', False):
            return
            
        # Only process once per batch (avoid duplicate processing)
        if getattr(p, '_freq_sep_batch_processed', False):
            return
        p._freq_sep_batch_processed = True
        
        # Get images tensor from kwargs
        images_tensor = kwargs.get('images')
        batch_number = kwargs.get('batch_number', 0)
        
        if images_tensor is None:
            print("⚠️ No images tensor found in postprocess_batch")
            return
            
        print(f"\n🎨 Frequency Separation: Processing batch {batch_number} before ADetailer...")
        print(f"   📊 Batch contains {images_tensor.shape[0]} images, shape: {images_tensor.shape}")
        
        try:
            # Get stored configuration
            freq_config = getattr(p, '_freq_sep_config')
            recombination_method = getattr(p, '_freq_sep_recombination_method')
            save_before_denoising = getattr(p, '_freq_sep_save_before_denoising', True)
            preserve_dc_component = getattr(p, '_freq_sep_preserve_dc_component', False)
            use_fft_shift = getattr(p, '_freq_sep_use_fft_shift', False)
            use_correct_fft_shift = getattr(p, '_freq_sep_use_correct_fft_shift', False)
            mask_function = getattr(p, '_freq_sep_mask_function', 'center_circular')
            
            # Convert tensor images to PIL and process each one
            enhanced_images = []
            for i in range(images_tensor.shape[0]):
                print(f"\n  🖼️ Processing image {i+1}/{images_tensor.shape[0]}...")
                
                # Convert tensor to PIL image
                image_tensor = images_tensor[i]  # Shape: [3, height, width]
                image_np = image_tensor.cpu().numpy()
                image_np = np.moveaxis(image_np, 0, 2)  # [H, W, 3]
                image_np = (image_np * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np)
                
                # Create a proper processing object
                temp_p = self._create_temp_processing_for_txt2img_simple(p, pil_image)
                
                # Process with frequency separation
                enhanced_image_list = self.process_latent_frequency_separation(
                    temp_p, freq_config, recombination_method, save_before_denoising, 
                    preserve_dc_component, use_fft_shift, use_correct_fft_shift, mask_function
                )
                
                if enhanced_image_list:
                    enhanced_images.append(enhanced_image_list[0])
                    print(f"   ✅ Image {i+1} frequency enhancement complete!")
                else:
                    enhanced_images.append(pil_image)
                    print(f"   ⚠️ Image {i+1} processing failed, keeping original")
            
            # Convert enhanced PIL images back to tensor format and update the batch
            enhanced_tensors = []
            device = images_tensor.device  # Use same device as original
            for enhanced_img in enhanced_images:
                # Convert PIL to numpy
                img_np = np.array(enhanced_img).astype(np.float32) / 255.0
                # Convert to tensor [H, W, 3] -> [3, H, W]
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(device)
                enhanced_tensors.append(img_tensor)
            
            # Stack back into batch tensor on same device
            enhanced_batch = torch.stack(enhanced_tensors).to(device)
            
            # Update the original tensor in-place
            images_tensor.copy_(enhanced_batch)
            print(f"   🔄 Updated original tensor in-place on device: {device}")
            
            print(f"\n🎉 Batch frequency enhancement complete! ADetailer will now process the enhanced images.")
            
        except Exception as e:
            print(f"❌ Error in batch frequency separation: {e}")
            import traceback
            traceback.print_exc()
            print("🔄 Keeping original images")
    
    def _create_temp_processing_for_txt2img_simple(self, original_p, image):
        """Create a processing object following ADetailer's approach"""
        from modules.processing import StableDiffusionProcessingImg2Img
        from copy import copy
        
        # Follow ADetailer's get_i2i_p method closely, only using attributes that exist
        temp_p = StableDiffusionProcessingImg2Img(
            init_images=[image],
            resize_mode=0,
            denoising_strength=0.5,  # Will be overridden per band
            mask=None,
            mask_blur=0,
            inpainting_fill=1,
            inpaint_full_res=False,
            inpaint_full_res_padding=0,
            inpainting_mask_invert=0,
            initial_noise_multiplier=None,
            sd_model=original_p.sd_model,
            outpath_samples=getattr(original_p, 'outpath_samples', 'outputs/img2img-images'),
            outpath_grids=getattr(original_p, 'outpath_grids', 'outputs/img2img-grids'),
            prompt=original_p.prompt,
            negative_prompt=original_p.negative_prompt,
            styles=getattr(original_p, 'styles', []),
            seed=original_p.seed,
            subseed=getattr(original_p, 'subseed', -1),
            subseed_strength=getattr(original_p, 'subseed_strength', 0),
            seed_resize_from_h=getattr(original_p, 'seed_resize_from_h', -1),
            seed_resize_from_w=getattr(original_p, 'seed_resize_from_w', -1),
            sampler_name=original_p.sampler_name,
            batch_size=1,
            n_iter=1,
            steps=original_p.steps,
            cfg_scale=original_p.cfg_scale,
            width=image.width,
            height=image.height,
            restore_faces=getattr(original_p, 'restore_faces', False),
            tiling=getattr(original_p, 'tiling', False),
            extra_generation_params=copy(getattr(original_p, 'extra_generation_params', {})),
            do_not_save_samples=True,
            do_not_save_grid=True,
            override_settings={}
        )
        
        # Following ADetailer's pattern exactly
        temp_p.cached_c = [None, None]
        temp_p.cached_uc = [None, None]
        
        # Add missing hr attributes to prevent infotext creation errors
        # These are needed for create_infotext() to work properly
        if not hasattr(temp_p, 'all_hr_prompts'):
            temp_p.all_hr_prompts = [temp_p.prompt]
        if not hasattr(temp_p, 'all_hr_negative_prompts'):
            temp_p.all_hr_negative_prompts = [temp_p.negative_prompt]
        
        # Copy scripts if available (like ADetailer does)
        if hasattr(original_p, 'scripts') and original_p.scripts is not None:
            temp_p.scripts = copy(original_p.scripts)
            temp_p.script_args = copy(getattr(original_p, 'script_args', []))
        
        # Prevent recursion - disable frequency separation for temp objects
        temp_p._frequency_separation_disabled = True
        temp_p._freq_sep_enabled = False  # Explicitly disable to prevent triggering postprocess_batch
        
        return temp_p
    
    def extra_params(self, p: StableDiffusionProcessing) -> dict:
        """Generate PNG metadata parameters for frequency separation settings"""
        
        # Only add metadata if frequency separation is enabled
        if not getattr(p, '_freq_sep_enabled', False):
            return {}
            
        params = {}
        
        # Main controls - NOTE: Don't store "enabled" flag as it causes infinite recursion
        # The presence of other FS parameters is sufficient to indicate it was used
        
        # Get stored configuration
        freq_config = getattr(p, '_freq_sep_config', None)
        if freq_config:
            params["Frequency Separation sync mode"] = freq_config.sync_mode.value
            params["Frequency Separation bands"] = freq_config.num_bands
            params["Frequency Separation overlap"] = freq_config.overlap_factor
            params["Frequency Separation spatial guidance"] = freq_config.spatial_guidance_strength
            
            # Band configurations
            for i, band_config in enumerate(freq_config.band_configs):
                band_name = ["low", "mid", "high"][i] if i < 3 else f"band{i+1}"
                params[f"Frequency Separation {band_name} freq start"] = band_config.frequency_range[0]
                params[f"Frequency Separation {band_name} freq end"] = band_config.frequency_range[1]
                params[f"Frequency Separation {band_name} denoising"] = band_config.denoising_strength
                params[f"Frequency Separation {band_name} amplitude"] = band_config.amplitude_scale
                params[f"Frequency Separation {band_name} steps"] = band_config.steps
                params[f"Frequency Separation {band_name} cfg"] = band_config.cfg_scale
        
        # Other settings
        recombination_method = getattr(p, '_freq_sep_recombination_method', None)
        if recombination_method:
            params["Frequency Separation recombination"] = recombination_method
            
        if getattr(p, '_freq_sep_save_before_denoising', False):
            params["Frequency Separation direct output"] = True
            
        if getattr(p, '_freq_sep_use_custom_steps_cfg', False):
            params["Frequency Separation custom steps cfg"] = True
            
        if getattr(p, '_freq_sep_preserve_dc_component', False):
            params["Frequency Separation preserve DC"] = True
            
        if getattr(p, '_freq_sep_use_fft_shift', False):
            params["Frequency Separation FFT shift"] = True
            
        if getattr(p, '_freq_sep_use_correct_fft_shift', False):
            params["Frequency Separation FFT shift detail"] = True
            
        mask_function = getattr(p, '_freq_sep_mask_function', None)
        if mask_function and mask_function != 'center_circular':  # Only store if not default
            params["Frequency Separation mask function"] = mask_function
        
        # Add version for future compatibility
        params["Frequency Separation version"] = __version__
        
        return params

# Frequency clamping callback for constraining frequencies during denoising
class FrequencyClampingCallback:
    def __init__(self):
        self.enabled = False
        self.frequency_mask = None
        self.band_name = None
        self.use_fft_shift = False
        
    def set_frequency_constraint(self, frequency_mask, band_name, use_fft_shift=False):
        """Set the frequency mask to apply during denoising"""
        self.enabled = True
        self.frequency_mask = frequency_mask
        self.band_name = band_name
        self.use_fft_shift = use_fft_shift
        
    def disable(self):
        """Disable frequency clamping"""
        self.enabled = False
        self.frequency_mask = None
        self.band_name = None
        
    def on_cfg_denoiser(self, params):
        """Apply frequency filtering to the denoised latent at each step"""
        if not self.enabled or self.frequency_mask is None:
            return
            
        try:
            # Get the denoised latent
            x = params.x
            
            # Convert to frequency domain
            x_freq = torch.fft.fftn(x, dim=(-2, -1))
            
            # Apply FFT shift if needed
            if self.use_fft_shift:
                x_freq = torch.fft.fftshift(x_freq, dim=(-2, -1))
            
            # Apply frequency mask (broadcast across batch and channels)
            mask_tensor = torch.tensor(self.frequency_mask, device=x.device, dtype=x.dtype)
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
            x_freq = x_freq * mask_tensor
            
            # Inverse FFT shift if applied
            if self.use_fft_shift:
                x_freq = torch.fft.ifftshift(x_freq, dim=(-2, -1))
            
            # Convert back to spatial domain
            x_filtered = torch.fft.ifftn(x_freq, dim=(-2, -1)).real
            
            # Update the denoised latent
            params.x = x_filtered
            
            # Debug print (only on first few steps to avoid spam)
            if params.sampling_step <= 2:
                print(f"        🎯 Applied frequency constraint for {self.band_name} at step {params.sampling_step}")
                
        except Exception as e:
            print(f"        ⚠️ Error in frequency clamping callback: {e}")
            # Don't break the denoising process

# Global instance of the callback
frequency_clamping_callback = FrequencyClampingCallback()

# Register the callback
script_callbacks.on_cfg_denoiser(frequency_clamping_callback.on_cfg_denoiser)

# Script registration - this is crucial for the script to actually work!
if DEPENDENCIES_AVAILABLE:
    # Register the real script
    script_instance = FrequencySeparationScript()
else:
    # Register dummy script if dependencies are missing
    script_instance = DummyScript()

# This ensures the script is properly registered with the WebUI
__all__ = ['FrequencySeparationScript', 'script_instance']