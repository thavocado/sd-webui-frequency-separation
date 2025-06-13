# Enhancement Features to Add Back to Working Version

## Overview
These features were implemented but broke the core processing. They should be added back **one at a time** to a working baseline, testing each addition carefully.

## Feature List

### 1. 💡 Preserve DC Component
**Purpose**: Special handling for zero-frequency (average brightness) component
**UI**: Checkbox with default `false`
**Implementation**: 
- During FFT processing, extract and preserve the DC component separately
- Apply special processing or preserve original DC value
- Restore during reconstruction to maintain overall brightness

### 2. 🎯 Use FFT Shift  
**Purpose**: Center DC component for proper frequency handling
**UI**: Checkbox with default `false`
**Implementation**:
- Apply `fftshift` after FFT to center low frequencies
- Apply `ifftshift` before inverse FFT
- Ensures proper frequency domain organization

### 3. 📊 Normalization Method
**Purpose**: How to preserve image brightness/energy after recombination
**UI**: Radio button with options:
- `"energy"` - Energy conservation (Parseval's theorem) 
- `"brightness"` - Simple brightness matching
- `"both"` - Apply both energy and brightness correction
- `"none"` - No normalization (original default)
**Default**: `"none"` (to preserve working behavior)

**Implementation**:
```python
if normalization_method == "energy" or normalization_method == "both":
    # Calculate energy ratio and apply sqrt scaling
    reconstructed_energy = np.mean(reconstructed_image ** 2)
    energy_ratio = total_original_energy / reconstructed_energy
    if 0.1 <= energy_ratio <= 10.0:  # Safety bounds
        reconstructed_image *= np.sqrt(energy_ratio)

if normalization_method == "brightness" or normalization_method == "both":
    # Simple brightness matching
    reconstructed_brightness = np.mean(reconstructed_image)
    brightness_ratio = original_brightness / reconstructed_brightness
    if 0.1 <= brightness_ratio <= 10.0:  # Safety bounds
        reconstructed_image *= brightness_ratio
```

### 4. 🌊 Preserve Phase Information (Optional)
**Purpose**: Maintain complex phase relationships during processing
**UI**: Checkbox with default `false`
**Status**: Currently non-functional due to diffusion model limitations
**Notes**: May need simpler single-FFT approach rather than double-FFT

## Critical Issues to Avoid

### ❌ Double-FFT Problem
**What went wrong**: The frequency reconstruction method was applying FFT to already-processed frequency band images, then masking them again, then doing inverse FFT. This double-processing broke the frequency separation.

**Correct approach**: Processed frequency bands should be combined directly (simple addition) since they're already the result of frequency-specific processing.

### ❌ Bypassed Normalization  
**What went wrong**: Early return in recombination method bypassed energy normalization code, causing overexposure.

**Correct approach**: Apply normalization after direct combination, before final output.

### ❌ Breaking Working Defaults
**What went wrong**: Changed defaults from working values to experimental values, breaking basic functionality.

**Correct approach**: Keep all enhancement features disabled by default, only enable as opt-in improvements.

## Implementation Strategy

1. **Start with working baseline** - Revert to last known working commit
2. **Add UI controls** - Add checkboxes/radio buttons with `false`/`"none"` defaults  
3. **Add one feature at a time**:
   - First: Add normalization options (easiest to test)
   - Second: Add FFT shift option (moderate complexity)
   - Third: Add DC component preservation (more complex)
   - Last: Phase preservation (if needed)
4. **Test each addition thoroughly** before moving to next feature
5. **Keep all features disabled by default** until proven stable

## Code Locations (for reference)

- **UI Definition**: Lines ~655-685 in frequency_separation.py
- **Parameter Storage**: Line ~857 (`p._freq_sep_*` attributes)
- **Parameter Retrieval**: Lines ~939+ in processing functions
- **Core Processing**: `_frequency_reconstruction_recombination` method
- **Energy Normalization**: Lines ~1773+ in recombination method

## Testing Protocol

For each feature addition:
1. Test with feature disabled (should work like baseline)
2. Test with feature enabled on simple image
3. Test with feature enabled on complex image  
4. Check for artifacts: overexposure, ringing, ghosting, haloing
5. Verify energy/brightness preservation
6. Only proceed to next feature if current one is stable