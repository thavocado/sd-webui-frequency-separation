# 🌊 Frequency Separation Extension for WebUI
*Unlock extra-sharp Stable Diffusion renders by processing low, mid & high frequencies separately*

![Frequency Separation Comparison](preview/comp.png)

---

## 1. Why You’d Want This  
**Motivation**  
Stable Diffusion’s VAE often softens tiny textures (skin pores, fabric weave, micro-contrast).  
This extension slices every picture into **three “layers” of detail** (structure, features, fine grain), runs each layer through its own diffusion pass, then glues them back together.  
The end result is visibly crisper and contrasty. Generated PNGs are usually **80 – 100 % larger** because there’s literally more information to compress.

**Technical detail**  
* We work in the Fourier domain. 

* Three radial masks select normalized bands  
  - **Low**  0 – 0.15 f<sub>max</sub>  (preserves composition)  
  - **Mid**  0.10 – 0.40  (edges & shapes)  
  - **High** 0.35 – 1.0  (textures)  
  Soft sigmoids with default 10 % overlap prevent ringing.

* Each band is inverse-FFT’d, diffused with custom steps/CFG, then re-FFT’d and merged.
---

## 2. Installation  
**Plain-English steps**  
1. `git clone` this repo in your WebUI `extensions/` or in **Extensions** tab "Install from URL": `https://github.com/thavocado/sd-webui-frequency-separation`
2. Restart the WebUI—look for the **“Frequency Separation”** accordion in *txt2img* & *img2img*.

## 3. Internal Pipeline

| Stage                  | Description  | Details                                                                                          |
| ---------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **FFT Split**          | Turn picture into a “heat-map of squiggles” and carve out low, mid, high zones like an audio EQ. | 2-D FFT, optional `fftshift`, radial / corner masks, DC preservation.                                   |
| **Per-Band Diffusion** | Each layer runs through SD with its own noise, CFG & steps.                 | Dynamically spawns `StableDiffusionProcessingImg2Img`; inherits prompt/seed; band-specific overrides.   |
| **Synchronization**    | Prevents ghosting between layers.                                               | Modes: independent, shared-noise, cross-attention, progressive, shared-latent.                          |
| **Reconstruction**     | Sharpened layers are re-stacked.                                                | Weighted spectrum merge or simple alpha blend → `ifft2`.                                                |
| **Batch Hook**         | Makes sure this all runs before ADetailer.                                  | `postprocess_batch` mutates the tensor batch in-place, so downstream scripts receive the enhanced data. |

---

## 4. Synchronization Modes

| Mode            | What is shared       |
| --------------- | -------------------- |
| Independent     | nothing              |
| Shared noise    | seed & ε<sub>t</sub> |
| Cross-attention | low-freq *c* vector  |
| Progressive     | latent chaining      |
| Shared latent   | mean latent + λ=0.3  |

---

## 5. Frequency Mask Functions

The extension offers 12 different mathematical distance functions for creating frequency masks. While experimental options are available for exploration, **corner_average** (default) works best in practice.

| Function | Description |
| --- | --- |
| **corner_average** ✅ | Average distance to all corners (default - recommended) |
| center_circular | Simple radial distance from center |
| corner_min_diamond | Minimum distance to any corner |
| corner_rms | Root mean square distance to corners |
| corner_harmonic | Harmonic mean distance to corners |
| corner_product | Product of normalized corner distances |
| transformed_circular | Circular mask with wraparound |
| manhattan | L1 norm distance |
| chebyshev | L∞ norm distance |
| minkowski_1_5 | Minkowski distance with p=1.5 |
| gravitational | Gravitational potential from corners |
| wave_interference | Wave interference from corner sources |

**Note:** Most experimental functions are included for research purposes. Stick with corner_average unless you have specific needs.

---

## 6. Advanced Band Configuration

Control each frequency band's processing independently for fine-tuned results.

### Default Band Ranges
- **Low Frequency**: 0.00 - 0.15 (Structure/Composition)
- **Mid Frequency**: 0.10 - 0.40 (Main Features)  
- **High Frequency**: 0.35 - 1.00 (Fine Details)

### Per-Band Controls
- **Denoising Strength**: Controls how much each frequency band is denoised during generation
  - Low: 0.3 (default)
  - Mid: 0.6 (default)
  - High: 0.8 (default)

- **Custom Steps & CFG** (optional): Enable "Use custom steps and CFG scale" to override global settings per band
  - Otherwise uses the main generation settings

---

## 7. Caveats

* Increases generation time by 3x. I recommend using it after you establish a base gen.
* No ComfyUI nodes (yet).
* Currently test on reForge only. Will work with other flavours of WebUI, but if you run into problems please let me know via https://github.com/thavocado/sd-webui-frequency-separation/issues.
