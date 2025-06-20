import gradio as gr
from typing import List, Tuple, Any
from freq_sep.config import SyncMode, FrequencyBandConfig, FreqSepConfig

def ui(is_img2img):
    """Create the UI components"""
    print(f"🎨 Frequency Separation UI: Creating interface (img2img={is_img2img})")
    
    # Show in both txt2img and img2img, but with different functionality  
    from modules.ui_components import InputAccordion
    
    with InputAccordion(value=False, label="🎛️ Frequency Separation Enhancement") as enabled:
            
        with gr.Row():
            sync_mode = gr.Dropdown(
                label="🔄 Synchronization Mode",
                info="How to coordinate processing between frequency bands",
                choices=[
                    SyncMode.INDEPENDENT.value,
                    SyncMode.SYNCHRONIZED_NOISE.value, 
                    SyncMode.CROSS_ATTENTION.value,
                    SyncMode.PROGRESSIVE.value,
                    SyncMode.SHARED_LATENT.value
                ],
                value=SyncMode.INDEPENDENT.value
            )
            
            num_bands = gr.Slider(
                label="📊 Number of Frequency Bands",
                info="Choose how many frequency bands to process (unused bands will be grayed out)",
                minimum=1, maximum=3, value=3, step=1
            )
        
        with gr.Row():
            overlap_factor = gr.Slider(
                label="🔗 Band Overlap Factor", 
                info="Prevents artifacts between frequency ranges",
                minimum=0.0, maximum=0.3, value=0.1, step=0.01
            )
            
            spatial_guidance = gr.Slider(
                label="🎯 Spatial Guidance Strength",
                info="Controls guidance strength for sync modes",
                minimum=0.0, maximum=1.0, value=0.3, step=0.05
            )
        
        with gr.Row():
            recombination_method = gr.Dropdown(
                label="🔄 Recombination Method",
                info="How to combine the processed frequency bands",
                choices=[
                    "frequency_reconstruction",
                    "simple_blending"
                ],
                value="frequency_reconstruction"
            )
            
            if is_img2img:
                save_before_denoising = gr.Checkbox(
                    label="💾 Output frequency-enhanced image directly",
                    info="Use the frequency-enhanced result as final output without additional img2img denoising",
                    value=False
                )
            else:
                # For txt2img, this is always the behavior
                save_before_denoising = gr.Checkbox(
                    label="💾 Apply frequency enhancement to generated image",
                    info="Enhance the generated image with frequency separation (always outputs directly)",
                    value=True,
                    visible=False  # Hidden because it's always true for txt2img
                )
        
        
        # All DC/FFT shift controls hidden for now - not ready for use
        # with gr.Row():
        #     preserve_dc_component_v2 = gr.Checkbox(
        #         label="💡 Preserve DC component",
        #         info="Maintain average brightness during frequency processing",
        #         value=False
        #     )
        #     use_fft_shift = gr.Checkbox(
        #         label="🔧 Use FFT Shift", 
        #         info="Center DC component with corrected frequency masks for better image quality",
        #         value=False
        #     )
        #     use_correct_fft_shift = gr.Checkbox(
        #         label="🎯 FFT Shift Detail Inspection",
        #         info="Inverted frequency masks - great for visualizing high-frequency details", 
        #         value=False
        #     )
        
        with gr.Row():
            mask_function = gr.Dropdown(
                label="🎭 Frequency Mask Function",
                choices=[
                    "no_mask",
                    "center_circular",
                    "corner_min_diamond", 
                    "corner_average",
                    "corner_rms",
                    "corner_harmonic",
                    "corner_product",
                    "transformed_circular",
                    "manhattan",
                    "chebyshev", 
                    "minkowski_1_5",
                    "gravitational",
                    "wave_interference"
                ],
                value="no_mask",
                info="Mathematical function for frequency separation masks"
            )
            
            latent_brightness_scale = gr.Slider(
                label="🔆 Brightness Scale",
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.05,
                info="Scale latent brightness (mean & std) before VAE decode"
            )
            
            enable_frequency_clamping = gr.Checkbox(
                label="🎯 Enable Frequency Clamping During Denoising",
                value=False,
                info="Constrain each band to its frequency range during denoising"
            )
        
        with gr.Row():
            second_order_filtering = gr.Checkbox(
                label="🌀 Second-Order Frequency Filtering",
                value=False,
                info="Apply frequency filtering to the image representation of the spectrum itself (more effective)"
            )
        
        with gr.Accordion("🎛️ Advanced Band Configuration", open=False):
            
            with gr.Row():
                use_custom_steps_cfg = gr.Checkbox(
                    label="🔧 Use custom steps and CFG scale",
                    info="Override the default generation settings with custom values per frequency band",
                    value=False
                )
            
            # Low Frequency Band
            with gr.Group() as low_freq_group:
                gr.HTML("<h4>🏗️ Low Frequency Band (Structure/Composition)</h4>")
                with gr.Row():
                    low_freq_range_start = gr.Slider(
                        label="Low Freq Start", minimum=0.0, maximum=1.0, value=0.0, step=0.01
                    )
                    low_freq_range_end = gr.Slider(
                        label="Low Freq End", minimum=0.0, maximum=1.0, value=0.45, step=0.01
                    )
                with gr.Row():
                    low_denoising = gr.Slider(
                        label="Denoising Strength", minimum=0.1, maximum=1.0, value=0.3, step=0.05
                    )
                    low_amplitude = gr.Slider(
                        label="Amplitude Scale", minimum=0.0, maximum=3.0, value=1.0, step=0.1
                    )
                with gr.Row():
                    low_steps = gr.Slider(
                        label="Steps", minimum=5, maximum=50, value=15, step=1, interactive=False
                    )
                    low_cfg = gr.Slider(
                        label="CFG Scale", minimum=1.0, maximum=20.0, value=5.0, step=0.5, interactive=False
                    )
            
            # Mid Frequency Band  
            with gr.Group() as mid_freq_group:
                gr.HTML("<h4>🎯 Mid Frequency Band (Main Features)</h4>")
                with gr.Row():
                    mid_freq_range_start = gr.Slider(
                        label="Mid Freq Start", minimum=0.0, maximum=1.0, value=0.45, step=0.01
                    )
                    mid_freq_range_end = gr.Slider(
                        label="Mid Freq End", minimum=0.0, maximum=1.0, value=0.75, step=0.01
                    )
                with gr.Row():
                    mid_denoising = gr.Slider(
                        label="Denoising Strength", minimum=0.1, maximum=1.0, value=0.6, step=0.05
                    )
                    mid_amplitude = gr.Slider(
                        label="Amplitude Scale", minimum=0.0, maximum=3.0, value=1.0, step=0.1
                    )
                with gr.Row():
                    mid_steps = gr.Slider(
                        label="Steps", minimum=5, maximum=50, value=20, step=1, interactive=False
                    )
                    mid_cfg = gr.Slider(
                        label="CFG Scale", minimum=1.0, maximum=20.0, value=7.0, step=0.5, interactive=False
                    )
            
            # High Frequency Band
            with gr.Group(visible=True) as high_freq_group:
                gr.HTML("<h4>✨ High Frequency Band (Fine Details)</h4>")
                with gr.Row():
                    high_freq_range_start = gr.Slider(
                        label="High Freq Start", minimum=0.0, maximum=1.0, value=0.75, step=0.01
                    )
                    high_freq_range_end = gr.Slider(
                        label="High Freq End", minimum=0.0, maximum=1.0, value=1.0, step=0.01
                    )
                with gr.Row():
                    high_denoising = gr.Slider(
                        label="Denoising Strength", minimum=0.1, maximum=1.0, value=0.8, step=0.05
                    )
                    high_amplitude = gr.Slider(
                        label="Amplitude Scale", minimum=0.0, maximum=3.0, value=1.0, step=0.1
                    )
                with gr.Row():
                    high_steps = gr.Slider(
                        label="Steps", minimum=5, maximum=50, value=25, step=1, interactive=False
                    )
                    high_cfg = gr.Slider(
                        label="CFG Scale", minimum=1.0, maximum=20.0, value=9.0, step=0.5, interactive=False
                    )
            
            # Enable/disable steps and CFG sliders based on checkbox state
            def toggle_custom_steps_cfg(enabled):
                return [
                    gr.update(interactive=enabled),  # low_steps
                    gr.update(interactive=enabled),  # low_cfg
                    gr.update(interactive=enabled),  # mid_steps
                    gr.update(interactive=enabled),  # mid_cfg
                    gr.update(interactive=enabled),  # high_steps
                    gr.update(interactive=enabled),  # high_cfg
                ]
            
            use_custom_steps_cfg.change(
                fn=toggle_custom_steps_cfg,
                inputs=[use_custom_steps_cfg],
                outputs=[low_steps, low_cfg, mid_steps, mid_cfg, high_steps, high_cfg]
            )
            
            # Enable/disable frequency band groups based on number of bands
            def toggle_frequency_bands(num_bands):
                updates = []
                for i, group in enumerate([low_freq_group, mid_freq_group, high_freq_group], 1):
                    enabled = i <= num_bands
                    updates.append(gr.update(visible=enabled))
                return updates
            
            num_bands.change(
                fn=toggle_frequency_bands,
                inputs=[num_bands],
                outputs=[low_freq_group, mid_freq_group, high_freq_group]
            )
    
    # No need for visibility logic with InputAccordion
    
    # Create hidden Gradio components for DC/FFT shift values (not ready for use)
    preserve_dc_component_v2 = gr.Checkbox(value=False, visible=False)  # Hidden - default off
    use_fft_shift = gr.Checkbox(value=False, visible=False)  # Hidden - default off
    use_correct_fft_shift = gr.Checkbox(value=False, visible=False)  # Hidden - default off
    
    # Set up infotext_fields for automatic metadata loading/applying
    infotext_fields = [
        # Main controls - NOTE: "enabled" field removed to prevent infinite recursion
        # Users can still manually enable the extension when pasting settings
        (sync_mode, "Frequency Separation sync mode"),
        (num_bands, "Frequency Separation bands"),
        (overlap_factor, "Frequency Separation overlap"),
        (spatial_guidance, "Frequency Separation spatial guidance"),
        (recombination_method, "Frequency Separation recombination"),
        (save_before_denoising, "Frequency Separation direct output"),
        (use_custom_steps_cfg, "Frequency Separation custom steps cfg"),
        (mask_function, "Frequency Separation mask function"),
        (latent_brightness_scale, "Frequency Separation brightness scale"),
        (enable_frequency_clamping, "Frequency Separation frequency clamping"),
        (second_order_filtering, "Frequency Separation second order filtering"),
        
        # Hidden controls (for future use)
        (preserve_dc_component_v2, "Frequency Separation preserve DC"),
        (use_fft_shift, "Frequency Separation FFT shift"), 
        (use_correct_fft_shift, "Frequency Separation FFT shift detail"),
        
        # Low frequency band
        (low_freq_range_start, "Frequency Separation low freq start"),
        (low_freq_range_end, "Frequency Separation low freq end"),
        (low_denoising, "Frequency Separation low denoising"),
        (low_amplitude, "Frequency Separation low amplitude"),
        (low_steps, "Frequency Separation low steps"),
        (low_cfg, "Frequency Separation low cfg"),
        
        # Mid frequency band  
        (mid_freq_range_start, "Frequency Separation mid freq start"),
        (mid_freq_range_end, "Frequency Separation mid freq end"),
        (mid_denoising, "Frequency Separation mid denoising"),
        (mid_amplitude, "Frequency Separation mid amplitude"),
        (mid_steps, "Frequency Separation mid steps"),
        (mid_cfg, "Frequency Separation mid cfg"),
        
        # High frequency band
        (high_freq_range_start, "Frequency Separation high freq start"),
        (high_freq_range_end, "Frequency Separation high freq end"),
        (high_denoising, "Frequency Separation high denoising"),
        (high_amplitude, "Frequency Separation high amplitude"),
        (high_steps, "Frequency Separation high steps"),
        (high_cfg, "Frequency Separation high cfg"),
    ]
    
    # Return both the UI components and the infotext_fields
    ui_components = [
        enabled, sync_mode, num_bands, overlap_factor, spatial_guidance, recombination_method,
        save_before_denoising, use_custom_steps_cfg, preserve_dc_component_v2, use_fft_shift, use_correct_fft_shift, mask_function,
        latent_brightness_scale, enable_frequency_clamping, second_order_filtering,
        low_freq_range_start, low_freq_range_end, low_denoising, low_amplitude, low_steps, low_cfg,
        mid_freq_range_start, mid_freq_range_end, mid_denoising, mid_amplitude, mid_steps, mid_cfg,
        high_freq_range_start, high_freq_range_end, high_denoising, high_amplitude, high_steps, high_cfg
    ]
    
    return ui_components, infotext_fields