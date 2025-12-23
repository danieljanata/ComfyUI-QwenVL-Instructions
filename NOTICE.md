# NOTICE

## Upstream project

This repository is a derivative work of **ComfyUI-QwenVL** ("upstream").

- Upstream: https://github.com/1038lab/ComfyUI-QwenVL
- Upstream license: **GNU General Public License v3.0 (or later)** (GPL-3.0-or-later)

Per the GPL, this repository is distributed under the same license. See `LICENSE`.

## What changed in this fork

This fork adds and exposes two nodes to make the workflow easier to maintain:

1. **QwenVL Model Downloader (Instructions)** – loads the Qwen-VL model once and outputs a `qwen_model` handle.
2. **Instructions (Instructions)** – accepts `qwen_model` + image/video + a free-form **Instructions** text input and outputs `RESPONSE`.

The original preset-prompt dropdown and the separate "custom prompt" field are not used by the new node.

## Trademarks

All upstream names and trademarks remain the property of their respective owners.
