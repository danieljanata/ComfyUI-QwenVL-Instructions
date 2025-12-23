"""ComfyUI-QwenVL - Instructions variant.

This repository is a derivative work of ComfyUI-QwenVL.

License: GNU GPL v3.0 or later (see LICENSE).
Attribution / change log: see NOTICE.

SPDX-License-Identifier: GPL-3.0-or-later
"""

import torch

from AILab_QwenVL import (
    QwenVLBase,
    HF_VL_MODELS,
    Quantization,
    ATTENTION_MODES,
    TOOLTIPS,
)

INSTR_TOOLTIP = (
    "Instruction text sent to Qwen-VL. This replaces any preset/custom prompt system. "
    "Write exactly how you want the model to describe or comment on the image/video."
)


class AILab_QwenVL_ModelDownloader(QwenVLBase):
    """
    Loads the selected Qwen-VL model once and outputs a reusable model handle.
    The handle can be connected to the 'Instructions' node to run inference.
    """

    @classmethod
    def INPUT_TYPES(cls):
        models = list(HF_VL_MODELS.keys())
        default_model = models[0] if models else "Qwen3-VL-4B-Instruct"
        return {
            "required": {
                "model_name": (models, {"default": default_model, "tooltip": TOOLTIPS.get("model_name", "")}),
                "quantization": (Quantization.get_values(), {"default": Quantization.FP16.value, "tooltip": TOOLTIPS.get("quantization", "")}),
                "attention_mode": (ATTENTION_MODES, {"default": "auto", "tooltip": TOOLTIPS.get("attention_mode", "")}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "tooltip": TOOLTIPS.get("keep_model_loaded", "")}),
                "use_torch_compile": ("BOOLEAN", {"default": False, "tooltip": TOOLTIPS.get("use_torch_compile", "")}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto", "tooltip": TOOLTIPS.get("device", "")}),
            }
        }

    RETURN_TYPES = ("QWENVL_MODEL",)
    RETURN_NAMES = ("qwen_model",)
    FUNCTION = "load"
    CATEGORY = "🧪AILab/QwenVL"

    def load(self, model_name, quantization, attention_mode, keep_model_loaded, use_torch_compile, device):
        # Ensure deterministic behavior for any internal ops (does not force generation determinism by itself).
        # We keep the model in this node instance so the next run can reuse it.
        self.load_model(
            model_name,
            quantization,
            attention_mode,
            use_torch_compile,
            device,
            keep_model_loaded,
        )
        return (self,)


class AILab_QwenVL_Instructions:
    """
    Runs Qwen-VL inference using a model handle (from Model Downloader) and a free-form instruction text.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_model": ("QWENVL_MODEL",),
                "Instructions": ("STRING", {"default": "Describe this image in detail.", "multiline": True, "tooltip": INSTR_TOOLTIP}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 2048, "tooltip": TOOLTIPS.get("max_tokens", "")}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31-1, "tooltip": TOOLTIPS.get("seed", "")}),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("RESPONSE",)
    FUNCTION = "process"
    CATEGORY = "🧪AILab/QwenVL"

    def process(self, qwen_model, Instructions, max_tokens, seed, image=None, video=None):
        # qwen_model is an instance of QwenVLBase returned from the downloader node.
        if qwen_model is None:
            raise ValueError("qwen_model is None. Connect 'QwenVL Model Downloader' to this node.")
        torch.manual_seed(int(seed))

        prompt = (Instructions or "").strip()
        if not prompt:
            prompt = "Describe this image in detail."

        # The original node uses: frame_count=8, temperature=0.2, top_p=1, num_beams=1, repetition_penalty=1.2
        text = qwen_model.generate(
            prompt_text=prompt,
            image=image,
            video=video,
            frame_count=8,
            max_tokens=int(max_tokens),
            temperature=0.2,
            top_p=1,
            num_beams=1,
            repetition_penalty=1.2,
        )
        return (text,)


NODE_CLASS_MAPPINGS = {
    "AILab_QwenVL_ModelDownloader": AILab_QwenVL_ModelDownloader,
    "AILab_QwenVL_Instructions": AILab_QwenVL_Instructions,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AILab_QwenVL_ModelDownloader": "QwenVL Model Downloader (Instructions)",
    "AILab_QwenVL_Instructions": "Instructions (Instructions)",
}
