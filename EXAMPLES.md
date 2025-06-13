# Frequency Separation Examples & Use Cases

## 🎯 Quick Start Examples

### Example 1: Portrait Enhancement
**Best for:** Portraits, faces, character art
```
Settings:
- Sync Mode: Progressive Refinement
- Low Freq (0.0-0.15): Denoising 0.2, Steps 12, CFG 5.0
- High Freq (0.3-1.0): Denoising 0.7, Steps 20, CFG 8.0
```
**Result:** Enhanced skin details while preserving facial structure

### Example 2: Landscape Detail Enhancement  
**Best for:** Nature scenes, architecture, detailed environments
```
Settings:
- Sync Mode: Cross-Attention Sync
- Low Freq (0.0-0.12): Denoising 0.3, Steps 15, CFG 6.0
- Mid Freq (0.1-0.4): Denoising 0.5, Steps 18, CFG 7.0
- High Freq (0.35-1.0): Denoising 0.8, Steps 25, CFG 9.0
```
**Result:** Crisp textures and fine details while maintaining composition

### Example 3: Speed-Optimized Processing
**Best for:** Quick iterations, testing concepts
```
Settings:
- Sync Mode: Synchronized Noise
- 2 Bands only
- Low Freq (0.0-0.25): Denoising 0.4, Steps 10, CFG 6.0
- High Freq (0.2-1.0): Denoising 0.6, Steps 15, CFG 7.5
```
**Result:** Faster processing with good quality improvement

## 🔧 Advanced Configuration Examples

### Heavy Detail Enhancement
**For maximum detail extraction:**
```
- Sync Mode: Progressive Refinement
- Bands: 4
- Band 1 (0.0-0.1): Denoising 0.2, Steps 10, CFG 4.0
- Band 2 (0.08-0.25): Denoising 0.4, Steps 15, CFG 6.0  
- Band 3 (0.2-0.5): Denoising 0.6, Steps 20, CFG 7.5
- Band 4 (0.45-1.0): Denoising 0.9, Steps 30, CFG 10.0
- Overlap Factor: 0.15
```

### Composition Preservation
**When you want to keep structure exactly as-is:**
```
- Sync Mode: Cross-Attention Sync
- Low Freq (0.0-0.2): Denoising 0.15, Steps 8, CFG 3.0
- High Freq (0.15-1.0): Denoising 0.8, Steps 25, CFG 9.0
- Spatial Guidance: 0.5
```

## 🎨 Use Case Scenarios

### Scenario 1: Upscaled Image Enhancement
**Problem:** AI upscaled images often lack fine detail
**Solution:** Use frequency separation to add realistic textures
```
Recommended: Progressive Refinement with heavy high-frequency denoising (0.8-0.9)
```

### Scenario 2: Anime/Cartoon Style Enhancement  
**Problem:** Need to enhance details without losing art style
**Solution:** Light processing on low frequencies, moderate on high
```
Recommended: Cross-Attention Sync with low denoising on structure (0.2-0.3)
```

### Scenario 3: Photorealistic Enhancement
**Problem:** Want to add fine skin/texture details to portraits
**Solution:** Aggressive high-frequency processing with structure preservation
```
Recommended: Progressive Refinement with strong high-freq processing
```

### Scenario 4: Artistic Style Transfer Refinement
**Problem:** Style transfer lost fine details from original
**Solution:** Frequency-aware reconstruction
```
Recommended: Shared Latent Processing to maintain style coherence
```

## ⚡ Performance Tips

### Memory Optimization
- Use 2-3 bands for lower VRAM usage
- Reduce steps on low frequency bands
- Use Independent mode for fastest processing

### Quality Optimization  
- Use Progressive Refinement for best results
- Increase overlap factor if you see band transition artifacts
- Higher spatial guidance for complex compositions

### Batch Processing
- Start with conservative settings
- Test on single images before batch runs
- Consider using presets for consistent results

## 🐛 Troubleshooting Common Issues

### Issue: Ghosting/Double Images
**Solution:** Increase spatial guidance, use Cross-Attention sync mode

### Issue: Processing Too Slow
**Solution:** Reduce bands to 2, use Synchronized Noise mode, lower steps

### Issue: Artifacts Between Frequency Bands
**Solution:** Increase overlap factor to 0.15-0.2

### Issue: Loss of Original Composition
**Solution:** Lower denoising on low frequencies, increase spatial guidance

### Issue: Not Enough Detail Enhancement
**Solution:** Increase high-frequency denoising and steps, try Progressive mode

## 📊 Recommended Presets by Content Type

| Content Type | Sync Mode | Bands | Low Freq Denoising | High Freq Denoising |
|--------------|-----------|-------|-------------------|-------------------|
| Portraits | Progressive | 3 | 0.2 | 0.7 |
| Landscapes | Cross-Attention | 3 | 0.3 | 0.8 |
| Anime/Art | Cross-Attention | 2 | 0.25 | 0.6 |
| Architecture | Progressive | 3 | 0.35 | 0.9 |
| Speed Test | Synchronized | 2 | 0.4 | 0.6 |

Remember: These are starting points! Adjust based on your specific image and desired outcome.