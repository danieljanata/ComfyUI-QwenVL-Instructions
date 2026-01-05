# NOTICE

## Upstream project

This repository is a derivative work of **ComfyUI-QwenVL** ("upstream").

- Upstream: https://github.com/1038lab/ComfyUI-QwenVL
- Upstream license: **GNU General Public License v3.0 (or later)** (GPL-3.0-or-later)

Per the GPL, this repository is distributed under the same license. See `LICENSE`.

## What changed in this fork (MultiSlot version)

This fork provides **three chainable nodes** for flexible workflow design:

### **1. QwenVL MultiSlot (5 IMG + Video)**

For processing multiple images and video:
- **5× IMAGE slots** (`image_1`..`image_5`) with per-slot seed and instructions
- **2× VIDEO inputs**: 
  - `video_frames` (IMAGE type) - for Load Image Sequence etc.
  - `video` (VIDEO type) - for VHS Video Loader etc.

### **2. QwenVL Text Enhancer**

For prompt enhancement (can be chained after MultiSlot):
- `text_input` - receives combined text (e.g., from Text Concatenate)
- `instructions` - how to enhance/modify the text
- `seed` - for reproducible results

### **3. QwenVL Single Image**

Simple node for single image processing:
- One image input
- Instructions via cable
- Minimal configuration

---

## Recommended Workflow (Chaining)

```
[Images] → [MultiSlot] → [Text Concatenate] → [TextEnhancer] → [Final Output]
              ↑                                      ↑
         instructions_1..5                      instructions
```

**Important**: Set `keep_model_loaded=TRUE` on MultiSlot so TextEnhancer reuses the loaded model!

---

## Key features:

1. **Instructions as cable inputs** - All instruction inputs use `forceInput: True` (no textarea widgets).

2. **Per-slot seed** - Each slot has its own seed for reproducible results.

3. **Video type support** - Both IMAGE batch and VIDEO type inputs supported.

4. **Chainable nodes** - Designed to work together without DAG cycle issues.

5. **Model sharing** - Use `keep_model_loaded=TRUE` to share model between nodes.

6. **Automatic slot ignore** - Empty slots are silently ignored.

---

## Technical changes:

- `AILab_QwenVL_Instructions.py`: Three new node classes
- VIDEO type support alongside IMAGE
- All inputs in `optional` with safe defaults
- Type-safe INT seeds
- Parameter clamping for safety

## Trademarks

All upstream names and trademarks remain the property of their respective owners.
