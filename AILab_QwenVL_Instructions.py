"""ComfyUI-QwenVL - Multi-Slot Instructions Nodes

This repository is a derivative work of ComfyUI-QwenVL.

License: GNU GPL v3.0 or later (see LICENSE).
Attribution / change log: see NOTICE.

SPDX-License-Identifier: GPL-3.0-or-later

This module provides TWO nodes that can be chained:
1. QwenVL MultiSlot (Images) - processes 5 images + 1 video
2. QwenVL Text Enhancer - processes text (can receive output from MultiSlot)

This allows: Images → MultiSlot → Concatenate → Text Enhancer → Final Output
"""

import random
import torch

from AILab_QwenVL import (
    QwenVLBase,
    HF_VL_MODELS,
    Quantization,
    ATTENTION_MODES,
    TOOLTIPS,
    clear_global_cache,
)


def _models_default():
    """Get available models and default selection."""
    models = list(HF_VL_MODELS.keys())
    default_model = models[0] if models else "Qwen3-VL-4B-Instruct"
    return models, default_model


def _clamp(value, min_val, max_val):
    """Clamp value to safe range."""
    return max(min_val, min(max_val, value))


def _safe_seed(seed_value):
    """
    Convert seed to safe integer. Handles None, empty, and out-of-range values.
    Returns a random seed if input is invalid.
    """
    if seed_value is None:
        return random.randint(0, 2**31 - 1)
    try:
        seed_int = int(seed_value)
        if seed_int < 0:
            return random.randint(0, 2**31 - 1)
        return seed_int % (2**31)
    except (ValueError, TypeError):
        return random.randint(0, 2**31 - 1)


def _get_device_options():
    """Get available device options."""
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_list = [f"cuda:{i}" for i in range(num_gpus)]
    return ["auto", "cpu", "mps"] + gpu_list


# =============================================================================
# NODE 1: QwenVL MultiSlot (Images + Video)
# =============================================================================

class AILab_QwenVL_MultiSlot_Images(QwenVLBase):
    """
    Multi-slot QwenVL node for processing images and video.
    
    - 5x IMAGE slots with per-slot seed and instructions
    - 1x VIDEO slot (accepts both IMAGE batch and VIDEO type)
    
    Instructions are STRING inputs via cable (forceInput: True).
    Slots without input or without instructions are ignored.
    
    Chain this with QwenVL_TextEnhancer for prompt enhancement.
    """

    @classmethod
    def INPUT_TYPES(cls):
        models, default_model = _models_default()
        device_options = _get_device_options()

        return {
            "required": {
                # === Global Model Settings ===
                "model_name": (models, {
                    "default": default_model,
                    "tooltip": TOOLTIPS.get("model_name", "Select the Qwen-VL model to use.")
                }),
                "quantization": (Quantization.get_values(), {
                    "default": Quantization.FP16.value,
                    "tooltip": TOOLTIPS.get("quantization", "Precision setting for memory vs quality tradeoff.")
                }),
                "device": (device_options, {
                    "default": "auto",
                    "tooltip": TOOLTIPS.get("device", "Device to run the model on.")
                }),
                "attention_mode": (ATTENTION_MODES, {
                    "default": "sdpa",
                    "tooltip": TOOLTIPS.get("attention_mode", "Attention implementation to use.")
                }),
                "use_torch_compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": TOOLTIPS.get("use_torch_compile", "Enable torch.compile for potential speedup.")
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": True,
                    "tooltip": TOOLTIPS.get("keep_model_loaded", "Keep model in VRAM between runs. Set TRUE to chain with TextEnhancer.")
                }),

                # === Global Generation Parameters ===
                "max_tokens": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "tooltip": TOOLTIPS.get("max_tokens", "Maximum tokens to generate.")
                }),
                "temperature": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": TOOLTIPS.get("temperature", "Sampling temperature.")
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": TOOLTIPS.get("top_p", "Nucleus sampling cutoff.")
                }),
                "num_beams": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "tooltip": TOOLTIPS.get("num_beams", "Beam search width.")
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": TOOLTIPS.get("repetition_penalty", "Penalty for repeated phrases.")
                }),
                "frame_count": ("INT", {
                    "default": 16,
                    "min": 1,
                    "max": 64,
                    "tooltip": TOOLTIPS.get("frame_count", "Number of frames to extract from video.")
                }),
            },
            "optional": {
                # === IMAGE SLOT 1 ===
                "image_1": ("IMAGE", {"tooltip": "First image input."}),
                "instructions_1": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Instructions for image_1 (connect via cable)."
                }),
                "seed_1": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**31 - 1,
                    "tooltip": "Seed for image_1 generation."
                }),

                # === IMAGE SLOT 2 ===
                "image_2": ("IMAGE", {"tooltip": "Second image input."}),
                "instructions_2": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Instructions for image_2 (connect via cable)."
                }),
                "seed_2": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**31 - 1,
                    "tooltip": "Seed for image_2 generation."
                }),

                # === IMAGE SLOT 3 ===
                "image_3": ("IMAGE", {"tooltip": "Third image input."}),
                "instructions_3": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Instructions for image_3 (connect via cable)."
                }),
                "seed_3": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**31 - 1,
                    "tooltip": "Seed for image_3 generation."
                }),

                # === IMAGE SLOT 4 ===
                "image_4": ("IMAGE", {"tooltip": "Fourth image input."}),
                "instructions_4": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Instructions for image_4 (connect via cable)."
                }),
                "seed_4": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**31 - 1,
                    "tooltip": "Seed for image_4 generation."
                }),

                # === IMAGE SLOT 5 ===
                "image_5": ("IMAGE", {"tooltip": "Fifth image input."}),
                "instructions_5": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Instructions for image_5 (connect via cable)."
                }),
                "seed_5": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**31 - 1,
                    "tooltip": "Seed for image_5 generation."
                }),

                # === VIDEO SLOT (IMAGE batch - e.g. from Load Image Sequence) ===
                "video_frames": ("IMAGE", {
                    "tooltip": "Video as IMAGE batch (from Load Image Sequence or similar)."
                }),
                
                # === VIDEO SLOT (VIDEO type - e.g. from VHS Video Loader) ===
                "video": ("VIDEO", {
                    "tooltip": "Video input (VIDEO type from video loaders like VHS)."
                }),
                
                "instructions_video": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Instructions for video (connect via cable)."
                }),
                "seed_video": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**31 - 1,
                    "tooltip": "Seed for video generation."
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("result_1", "result_2", "result_3", "result_4", "result_5", "result_video")
    FUNCTION = "run"
    CATEGORY = "QwenVL (MultiSlot)"

    def _process_image_slot(
        self,
        image,
        instructions,
        seed,
        max_tokens,
        temperature,
        top_p,
        num_beams,
        repetition_penalty,
    ):
        """Process a single image slot with its own seed."""
        if image is None:
            return ""
        
        prompt = (instructions or "").strip()
        if not prompt:
            return ""

        torch.manual_seed(_safe_seed(seed))

        text = self.generate(
            prompt,
            image,
            None,  # video
            0,     # frame_count
            int(_clamp(max_tokens, 1, 8192)),
            float(_clamp(temperature, 0.0, 2.0)),
            float(_clamp(top_p, 0.0, 1.0)),
            int(_clamp(num_beams, 1, 8)),
            float(_clamp(repetition_penalty, 0.5, 2.0)),
        )
        return text

    def _process_video_slot(
        self,
        video_data,
        instructions,
        seed,
        frame_count,
        max_tokens,
        temperature,
        top_p,
        num_beams,
        repetition_penalty,
    ):
        """Process video slot with its own seed."""
        if video_data is None:
            return ""
        
        prompt = (instructions or "").strip()
        if not prompt:
            return ""

        torch.manual_seed(_safe_seed(seed))

        text = self.generate(
            prompt,
            None,  # image
            video_data,
            int(_clamp(frame_count, 1, 64)),
            int(_clamp(max_tokens, 1, 8192)),
            float(_clamp(temperature, 0.0, 2.0)),
            float(_clamp(top_p, 0.0, 1.0)),
            int(_clamp(num_beams, 1, 8)),
            float(_clamp(repetition_penalty, 0.5, 2.0)),
        )
        return text

    def run(
        self,
        # === Global Model Settings ===
        model_name,
        quantization,
        device,
        attention_mode,
        use_torch_compile,
        keep_model_loaded,
        # === Global Generation Parameters ===
        max_tokens,
        temperature,
        top_p,
        num_beams,
        repetition_penalty,
        frame_count,
        # === Image Slots (all optional) ===
        image_1=None,
        instructions_1=None,
        seed_1=0,
        image_2=None,
        instructions_2=None,
        seed_2=0,
        image_3=None,
        instructions_3=None,
        seed_3=0,
        image_4=None,
        instructions_4=None,
        seed_4=0,
        image_5=None,
        instructions_5=None,
        seed_5=0,
        # === Video Slots (optional) ===
        video_frames=None,
        video=None,
        instructions_video=None,
        seed_video=0,
    ):
        """Process all connected image and video slots."""
        
        # Load model once
        self.load_model(
            model_name,
            quantization,
            attention_mode,
            bool(use_torch_compile),
            device,
            bool(keep_model_loaded),
        )

        # Initialize outputs
        result_1 = ""
        result_2 = ""
        result_3 = ""
        result_4 = ""
        result_5 = ""
        result_video = ""

        try:
            # Process IMAGE slots
            result_1 = self._process_image_slot(
                image_1, instructions_1, seed_1,
                max_tokens, temperature, top_p, num_beams, repetition_penalty
            )
            result_2 = self._process_image_slot(
                image_2, instructions_2, seed_2,
                max_tokens, temperature, top_p, num_beams, repetition_penalty
            )
            result_3 = self._process_image_slot(
                image_3, instructions_3, seed_3,
                max_tokens, temperature, top_p, num_beams, repetition_penalty
            )
            result_4 = self._process_image_slot(
                image_4, instructions_4, seed_4,
                max_tokens, temperature, top_p, num_beams, repetition_penalty
            )
            result_5 = self._process_image_slot(
                image_5, instructions_5, seed_5,
                max_tokens, temperature, top_p, num_beams, repetition_penalty
            )

            # Process VIDEO slot - prefer video_frames (IMAGE), fallback to video (VIDEO)
            # Handle VIDEO type - extract frames if needed
            video_data = None
            if video_frames is not None:
                video_data = video_frames
            elif video is not None:
                # VIDEO type from VHS etc. - might be dict with 'images' key or tensor
                if isinstance(video, dict):
                    video_data = video.get("images", video.get("frames", None))
                else:
                    video_data = video
            
            if video_data is not None:
                result_video = self._process_video_slot(
                    video_data, instructions_video, seed_video,
                    frame_count, max_tokens, temperature, top_p, num_beams, repetition_penalty
                )

            return (result_1, result_2, result_3, result_4, result_5, result_video)

        finally:
            if not keep_model_loaded:
                self.clear()


# =============================================================================
# NODE 2: QwenVL Text Enhancer
# =============================================================================

class AILab_QwenVL_TextEnhancer(QwenVLBase):
    """
    Text enhancement node using QwenVL.
    
    Takes text input and instructions, outputs enhanced/modified text.
    Can be chained after MultiSlot_Images to enhance combined prompts.
    
    Use keep_model_loaded=True on the previous node to share the model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        models, default_model = _models_default()
        device_options = _get_device_options()

        return {
            "required": {
                # === Global Model Settings ===
                "model_name": (models, {
                    "default": default_model,
                    "tooltip": TOOLTIPS.get("model_name", "Select the Qwen-VL model.")
                }),
                "quantization": (Quantization.get_values(), {
                    "default": Quantization.FP16.value,
                    "tooltip": TOOLTIPS.get("quantization", "Precision setting.")
                }),
                "device": (device_options, {
                    "default": "auto",
                    "tooltip": TOOLTIPS.get("device", "Device to run the model on.")
                }),
                "attention_mode": (ATTENTION_MODES, {
                    "default": "sdpa",
                    "tooltip": TOOLTIPS.get("attention_mode", "Attention implementation.")
                }),
                "use_torch_compile": ("BOOLEAN", {
                    "default": False,
                    "tooltip": TOOLTIPS.get("use_torch_compile", "Enable torch.compile.")
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Keep model loaded after this node. Set FALSE if this is the last QwenVL node."
                }),

                # === Generation Parameters ===
                "max_tokens": ("INT", {
                    "default": 512,
                    "min": 1,
                    "max": 8192,
                    "tooltip": TOOLTIPS.get("max_tokens", "Maximum tokens to generate.")
                }),
                "temperature": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": TOOLTIPS.get("temperature", "Sampling temperature.")
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": TOOLTIPS.get("top_p", "Nucleus sampling cutoff.")
                }),
                "num_beams": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 8,
                    "tooltip": TOOLTIPS.get("num_beams", "Beam search width.")
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.2,
                    "min": 0.5,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": TOOLTIPS.get("repetition_penalty", "Penalty for repeated phrases.")
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 2**31 - 1,
                    "tooltip": "Seed for generation."
                }),
            },
            "optional": {
                # === Text Input ===
                "text_input": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Text to enhance/modify (connect from Concatenate or other text source)."
                }),
                "instructions": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Instructions for text modification (connect via cable)."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_text",)
    FUNCTION = "run"
    CATEGORY = "QwenVL (MultiSlot)"

    def run(
        self,
        # === Model Settings ===
        model_name,
        quantization,
        device,
        attention_mode,
        use_torch_compile,
        keep_model_loaded,
        # === Generation Parameters ===
        max_tokens,
        temperature,
        top_p,
        num_beams,
        repetition_penalty,
        seed,
        # === Text Input (optional) ===
        text_input=None,
        instructions=None,
    ):
        """Process text input with instructions."""
        
        # Check if we have valid input
        if not text_input or not text_input.strip():
            return ("",)
        
        instr = (instructions or "").strip()
        if not instr:
            # No instructions - just return the input unchanged
            return (text_input,)
        
        # Load model
        self.load_model(
            model_name,
            quantization,
            attention_mode,
            bool(use_torch_compile),
            device,
            bool(keep_model_loaded),
        )

        try:
            # Combine instructions with text
            prompt = f"{instr}\n\nText to process:\n{text_input.strip()}"

            torch.manual_seed(_safe_seed(seed))

            result = self.generate(
                prompt,
                None,  # image
                None,  # video
                0,     # frame_count
                int(_clamp(max_tokens, 1, 8192)),
                float(_clamp(temperature, 0.0, 2.0)),
                float(_clamp(top_p, 0.0, 1.0)),
                int(_clamp(num_beams, 1, 8)),
                float(_clamp(repetition_penalty, 0.5, 2.0)),
            )
            return (result,)

        finally:
            if not keep_model_loaded:
                self.clear()


# =============================================================================
# NODE 3: QwenVL Single Image (simple node for single image processing)
# =============================================================================

class AILab_QwenVL_SingleImage(QwenVLBase):
    """
    Simple single-image QwenVL node.
    
    For users who just need to process one image with instructions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        models, default_model = _models_default()
        device_options = _get_device_options()

        return {
            "required": {
                "model_name": (models, {"default": default_model}),
                "quantization": (Quantization.get_values(), {"default": Quantization.FP16.value}),
                "device": (device_options, {"default": "auto"}),
                "attention_mode": (ATTENTION_MODES, {"default": "sdpa"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "num_beams": ("INT", {"default": 1, "min": 1, "max": 8}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Image to analyze."}),
                "instructions": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Instructions for image analysis (connect via cable)."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "run"
    CATEGORY = "QwenVL (MultiSlot)"

    def run(
        self,
        model_name,
        quantization,
        device,
        attention_mode,
        keep_model_loaded,
        max_tokens,
        temperature,
        top_p,
        num_beams,
        repetition_penalty,
        seed,
        image=None,
        instructions=None,
    ):
        if image is None:
            return ("",)
        
        prompt = (instructions or "").strip()
        if not prompt:
            return ("",)

        self.load_model(
            model_name,
            quantization,
            attention_mode,
            False,  # use_torch_compile
            device,
            bool(keep_model_loaded),
        )

        try:
            torch.manual_seed(_safe_seed(seed))
            
            result = self.generate(
                prompt,
                image,
                None, 0,
                int(_clamp(max_tokens, 1, 8192)),
                float(_clamp(temperature, 0.0, 2.0)),
                float(_clamp(top_p, 0.0, 1.0)),
                int(_clamp(num_beams, 1, 8)),
                float(_clamp(repetition_penalty, 0.5, 2.0)),
            )
            return (result,)
        finally:
            if not keep_model_loaded:
                self.clear()


# =============================================================================
# NODE 4: QwenVL Model Offloader (for manual cache clearing)
# =============================================================================

class AILab_QwenVL_Offloader:
    """
    Utility node to manually clear the QwenVL model from VRAM/RAM.
    
    Connect any STRING output to this node's trigger input to ensure it runs
    at the end of your workflow and frees memory.
    
    Alternative: Set keep_model_loaded=FALSE on the last QwenVL node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "trigger_string": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Connect any STRING output here to trigger offload."
                }),
                "trigger_image": ("IMAGE", {
                    "tooltip": "Connect any IMAGE output here to trigger offload."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "run"
    CATEGORY = "QwenVL (MultiSlot)"
    OUTPUT_NODE = True

    def run(self, trigger_string=None, trigger_image=None):
        """Clear the global model cache."""
        clear_global_cache()
        return ("QwenVL model offloaded from VRAM",)


# =============================================================================
# Node Registration
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "AILab_QwenVL_MultiSlot_Images": AILab_QwenVL_MultiSlot_Images,
    "AILab_QwenVL_TextEnhancer": AILab_QwenVL_TextEnhancer,
    "AILab_QwenVL_SingleImage": AILab_QwenVL_SingleImage,
    "AILab_QwenVL_Offloader": AILab_QwenVL_Offloader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_QwenVL_MultiSlot_Images": "QwenVL MultiSlot (5 IMG + Video)",
    "AILab_QwenVL_TextEnhancer": "QwenVL Text Enhancer",
    "AILab_QwenVL_SingleImage": "QwenVL Single Image",
    "AILab_QwenVL_Offloader": "QwenVL Model Offloader",
}
