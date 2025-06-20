from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union

class SyncMode(Enum):
    INDEPENDENT = "independent"
    SYNCHRONIZED_NOISE = "synchronized_noise"
    CROSS_ATTENTION = "cross_attention"
    PROGRESSIVE = "progressive_refinement"
    SHARED_LATENT = "shared_latent"


@dataclass
class FrequencyBandConfig:
    """Configuration for processing a specific frequency band"""
    name: str
    frequency_range: Tuple[float, float]  # (low_freq, high_freq) in normalized units
    denoising_strength: float
    steps: int
    cfg_scale: float
    amplitude_scale: float = 1.0  # Amplitude scaling factor for recombination
    preserve_composition: bool = False


@dataclass
class FreqSepConfig:
    """Main configuration for frequency separation processing"""
    sync_mode: SyncMode = SyncMode.INDEPENDENT
    num_bands: int = 3
    band_configs: List[FrequencyBandConfig] = None
    overlap_factor: float = 0.1  # Frequency band overlap to prevent artifacts
    spatial_guidance_strength: float = 0.3  # For cross-attention and shared latent modes
    
    def __post_init__(self):
        if self.band_configs is None:
            self.band_configs = self._create_default_bands()
    
    def _create_default_bands(self) -> List[FrequencyBandConfig]:
        """Create default 3-band configuration"""
        return [
            FrequencyBandConfig(
                name="low_freq",
                frequency_range=(0.0, 0.15),
                denoising_strength=0.3,  # Light denoising to preserve composition
                steps=15,
                cfg_scale=5.0,
                preserve_composition=True
            ),
            FrequencyBandConfig(
                name="mid_freq", 
                frequency_range=(0.1, 0.4),
                denoising_strength=0.6,  # Moderate denoising for features
                steps=20,
                cfg_scale=7.0
            ),
            FrequencyBandConfig(
                name="high_freq",
                frequency_range=(0.35, 1.0),
                denoising_strength=0.8,  # Heavy denoising for detail enhancement
                steps=25,
                cfg_scale=9.0
            )
        ]
