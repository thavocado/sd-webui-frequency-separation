# Frequency Separation Enhancement Extension

An advanced extension for Stable Diffusion WebUI reForge that overcomes VAE limitations by processing different frequency bands of images independently through the diffusion process.

## Core Innovation

The VAE (Variational Autoencoder) in Stable Diffusion has channel limitations that prevent it from producing fine details. This extension addresses this by:

1. **Frequency Band Separation**: Uses Fast Fourier Transform (FFT) to split images into frequency bands:
   - **Low frequencies**: Overall structure and composition
   - **Mid frequencies**: Main features and objects  
   - **High frequencies**: Fine details and textures

2. **Optimized Processing**: Each frequency band is processed with different optimal diffusion settings:
   - Low frequencies: Light denoising to preserve composition
   - High frequencies: Heavy denoising to enhance details

3. **Mathematical Recombination**: Processed bands are mathematically recombined for superior quality beyond normal VAE limitations

## Synchronization Modes

To prevent ghosting and spatial misalignment issues when processing frequency bands separately:

### Independent Processing
- Original behavior with potential ghosting
- Each band processed with separate random seeds
- Fastest but may have artifacts

### Synchronized Noise
- Uses the same random seed across all frequency bands
- Maintains spatial coherence while allowing band-specific processing
- Good balance of quality and speed

### Cross-Attention Synchronization  
- Low frequency result guides processing of higher frequencies
- Ensures structural consistency across bands
- Excellent for preserving composition while enhancing details

### Progressive Refinement
- Processes bands sequentially from low to high frequency
- Each stage builds on the previous results
- Best quality but slower processing

### Shared Latent Processing
- Mixes frequency bands with spatial guidance during processing
- Maintains coherence through shared latent representations
- Good for complex images with lots of detail

## Usage

1. Go to the **img2img** tab
2. Load your input image  
3. Scroll down to find the **"🎛️ Frequency Separation Enhancement"** section (it appears as its own dedicated section, not in the script dropdown)
4. Check **"Enable Frequency Separation"** 
5. The advanced controls will appear - choose your preferred synchronization mode
6. Adjust frequency band settings if needed (defaults work well for most images)
7. Generate enhanced images!

## Parameters

### Main Controls
- **Synchronization Mode**: Choose processing approach
- **Number of Frequency Bands**: 2-5 bands (3 recommended)
- **Band Overlap Factor**: Prevents artifacts between bands
- **Spatial Guidance Strength**: Controls guidance strength for sync modes

### Per-Band Configuration
Each frequency band can be configured with:
- **Frequency Range**: Which frequencies this band covers
- **Denoising Strength**: How much to denoise this band
- **Steps**: Number of diffusion steps for this band
- **CFG Scale**: Classifier-free guidance scale

## Recommended Settings

### For Maximum Detail Enhancement
- **Sync Mode**: Progressive Refinement
- **Low Freq**: 0.0-0.15, Denoising 0.3, Steps 15, CFG 5.0
- **Mid Freq**: 0.1-0.4, Denoising 0.6, Steps 20, CFG 7.0  
- **High Freq**: 0.35-1.0, Denoising 0.8, Steps 25, CFG 9.0

### For Balanced Quality/Speed
- **Sync Mode**: Cross-Attention Synchronization
- **Low Freq**: 0.0-0.2, Denoising 0.4, Steps 12, CFG 6.0
- **High Freq**: 0.15-1.0, Denoising 0.7, Steps 18, CFG 8.0

### For Fast Processing
- **Sync Mode**: Synchronized Noise  
- Use 2 bands with moderate settings

## Technical Details

The extension uses PyTorch FFT operations for frequency domain processing and implements sophisticated blending algorithms to recombine processed bands while maintaining spatial coherence.

**Requirements**: 
- Stable Diffusion WebUI reForge
- PyTorch with CUDA support recommended
- Sufficient VRAM for processing multiple bands

## Tips

- Start with default settings and adjust based on your image content
- Higher frequency bands benefit from more denoising steps
- Use progressive refinement for maximum quality on important images
- Reduce band overlap if you see artifacts between frequency ranges
- Increase spatial guidance for better coherence in complex images

## Troubleshooting

**Memory Issues**: Reduce number of bands or processing steps  
**Artifacts**: Increase band overlap factor or try different sync mode  
**Slow Processing**: Use fewer bands or synchronized noise mode  
**Poor Results**: Adjust frequency ranges for your specific image content

This extension pushes the boundaries of what's possible with Stable Diffusion by bypassing VAE limitations through intelligent frequency domain processing.

## 🐛 Debugging & Troubleshooting

### Verifying the Extension is Working

The extension includes extensive debug output to help track its operation:

1. **Startup Messages**: Look for `🌊 Frequency Separation Extension: Initialized successfully!` in the console
2. **UI Creation**: When switching to img2img, you should see `🎨 Frequency Separation UI: Creating interface`
3. **Processing Messages**: When generating with the extension enabled, look for:
   - `🌊 Frequency Separation: Starting latent-space frequency processing...`
   - `🧠 Processing X images in latent space with [sync_mode] sync mode`
   - `🎯 Processing image X/Y in latent space...`
   - `📐 Encoded to latent shape: [shape]`
   - `🌊 Split into X frequency bands`
   - `🎵 [band_name]: [freq_range] -> [latent_shape]`
   - `🎛️ Processing [band] band (denoising: X, steps: Y)`
   - `🔄 Recombined latent shape: [shape]`
   - `🖼️ Decoding latent X/Y to final image...`
   - `🎉 Latent-space frequency separation complete!`

### Current Implementation Status

**Version 2.0 - Latent Space Processing**: The extension now properly implements **latent-space frequency separation** which is the correct approach for defeating VAE limitations.

**What it does**:
1. **Encodes images to latent space** using the model's VAE encoder
2. **Applies FFT-based frequency separation** directly in the latent domain
3. **Processes each frequency band** with different enhancement parameters
4. **Recombines frequency bands** mathematically in latent space
5. **Decodes the final result** using the VAE decoder

**Why this matters**:
- Working in latent space bypasses VAE encoding/decoding losses
- Frequency separation in latent domain preserves model-specific representations  
- Each band can theoretically be processed through different diffusion parameters
- This is the correct architectural approach for the intended functionality

### Recent Major Update (v2.0)

**Moved to Latent Space Processing**:
- **Issue**: Previous version worked on decoded images, missing the core innovation
- **Fix**: Complete rewrite to work directly in VAE latent space
- **Result**: True frequency separation that bypasses VAE limitations

**Architecture Changes**:
- ✅ **VAE Encode**: Input images → latent tensors  
- ✅ **FFT Processing**: Frequency separation in latent domain
- ✅ **Band Enhancement**: Different processing per frequency band
- ✅ **Latent Recombination**: Mathematical frequency domain recombination
- ✅ **VAE Decode**: Final latent → enhanced output image

**What You Should See Now**:
- Detailed latent processing messages showing encode → process → decode flow
- Latent tensor shapes and frequency band information
- Processing happening in the correct latent space domain

### Common Issues

**Extension Not Appearing**: 
- Ensure you restarted WebUI after installing
- Check console for initialization messages
- Verify you're in img2img mode (extension only works there)

**No Processing Messages**:
- Make sure "Enable Frequency Separation" is checked
- Verify you have input images loaded
- Check console for error messages

**Processing But No Effect**:
- Look for the latent processing messages showing encode → process → decode flow
- Check that latent tensor shapes are being reported (e.g., `[1, 4, 64, 64]`)
- Ensure you see frequency band split messages with different frequency ranges
- Verify the final `🎉 Latent-space frequency separation complete!` message

**Dependencies Missing**:
- The extension will show a warning if required packages are missing
- Run the install.py script or manually install requirements