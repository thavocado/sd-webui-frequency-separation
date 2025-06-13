# 🌊 Frequency Separation Extension for WebUI
*Unlock extra-sharp Stable Diffusion renders by processing low, mid & high frequencies separately*

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
1. Put this repo in `extensions/`.  
2. Restart the WebUI—look for the **“Frequency Separation”** accordion in *txt2img* & *img2img*.

```bash
git clone https://github.com/yourname/sd-webui-frequency-separation \
      ~/.stable-diffusion-webui/extensions/frequency-separation
````

## 3. Internal Pipeline

| Stage                  | Description  | Details                                                                                          |
| ---------------------- | ------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **FFT Split**          | Turn picture into a “heat-map of squiggles” and carve out low, mid, high zones line an audio EQ. | 2-D FFT, optional `fftshift`, radial / corner masks, DC preservation.                                   |
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

## 5. Caveats

* Increases generation time by 3x. I recommend using it after you establish a base gen.
