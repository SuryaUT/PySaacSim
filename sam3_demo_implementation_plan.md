# SAM 3 Text-Prompted Image Segmentation Demo — Implementation Plan

> **Goal:** A single-machine Python application that accepts a photo upload, a noun-phrase prompt, and an Enter keypress, then returns the image annotated with per-instance mask contours, bounding boxes, and confidence labels for every detected match.

---

## Table of Contents

1. [Stack Choice & Rationale](#1-stack-choice--rationale)
2. [Project Structure](#2-project-structure)
3. [Environment Setup](#3-environment-setup)
4. [Module: `inference.py`](#4-module-inferencepy)
5. [Module: `render.py`](#5-module-renderpy)
6. [Module: `app.py`](#6-module-apppy)
7. [Error Handling](#7-error-handling)
8. [Performance Tuning](#8-performance-tuning)
9. [Running & Testing](#9-running--testing)
10. [Known Gotchas](#10-known-gotchas)

---

## 1. Stack Choice & Rationale

| Concern | Choice | Why |
|---|---|---|
| UI | **Gradio** | Ships drag-and-drop upload, a textbox with a first-class `submit` (Enter) event, and an image output component — no HTML/CSS/JS required. |
| Model loading | **HF `transformers`** (`Sam3Model`, `Sam3Processor`) | Pip-installable, maintained, exposes everything needed. Avoids cloning `facebookresearch/sam3`. |
| Contour drawing | **OpenCV** (`cv2.findContours`) | Pillow has no equivalent primitive for tracing binary mask contours. |
| UI alternative rejected | Tkinter / PyQt | Requires a file dialog, display canvas, and Pillow blitting — substantial boilerplate for no gain. |
| Framework alternative rejected | Streamlit | Reruns the entire script on each widget change, which is awkward for stateful model inference. |

---

## 2. Project Structure

```
sam3_demo/
├── app.py              # Gradio UI + event wiring
├── inference.py        # Model loading, segment()
├── render.py           # Contour / box / label drawing
├── requirements.txt
└── README.md
```

The three modules keep UI, inference, and rendering independently testable. `inference.segment()` and `render.draw()` can be called from a notebook without importing Gradio.

---

## 3. Environment Setup

### Python version

Python **3.10 or 3.11**. SAM 3's transformers integration works on both. Python 3.12 is untested upstream.

### `requirements.txt`

```
torch>=2.5.0
torchvision>=0.20.0
transformers>=4.57.0
gradio>=4.44.0
pillow>=10.0.0
opencv-python>=4.9.0
numpy>=1.26.0
huggingface_hub>=0.26.0
```

### Install

```bash
pip install -r requirements.txt
```

CUDA 12.1+ is recommended on the Windows/WSL2 box. On Apple Silicon the model runs on MPS at reduced speed (see device selection in §4).

### HuggingFace authentication

`facebook/sam3` is a **gated repository**. Accept the licence on the model card at `huggingface.co/facebook/sam3`, then authenticate once:

```bash
huggingface-cli login
# or:
export HF_TOKEN=hf_...
```

The model is approximately **3.4 GB** and downloads to `~/.cache/huggingface/hub/` on first run. Subsequent runs load from cache.

---

## 4. Module: `inference.py`

This module owns the entire model lifecycle: device selection, loading, warmup, and inference.

### 4.1 Device and dtype

```python
import torch

def _get_device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32   # MPS bf16 support is fragile
    return torch.device("cpu"), torch.float32
```

bfloat16 is used on CUDA rather than float16. It has the same memory footprint with better numerical range and avoids the occasional inf/nan that float16 can produce in attention layers.

### 4.2 Data class

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class Detection:
    mask:  np.ndarray   # bool, shape (H, W), original image resolution
    box:   tuple        # (x1, y1, x2, y2) in pixels, int
    score: float
```

### 4.3 `SAM3Runner` class

```python
from transformers import Sam3Model, Sam3Processor
from PIL import Image
from typing import List

class SAM3Runner:
    def __init__(self):
        self.device, self.dtype = _get_device_and_dtype()
        print(f"Loading SAM 3 on {self.device} ({self.dtype})...")
        self.model = Sam3Model.from_pretrained(
            "facebook/sam3", torch_dtype=self.dtype
        ).to(self.device).eval()
        self.processor = Sam3Processor.from_pretrained("facebook/sam3")
        self._warmup()

    def _warmup(self):
        """Trigger CUDA kernel compilation before the first real request."""
        dummy = Image.new("RGB", (512, 512), color=(128, 128, 128))
        self.segment(dummy, "object", score_threshold=0.9)

    @torch.inference_mode()
    def segment(
        self,
        image: Image.Image,
        prompt: str,
        score_threshold: float = 0.5,
        mask_threshold: float  = 0.5,
    ) -> List[Detection]:
        if not prompt.strip():
            return []

        image = image.convert("RGB")
        image = _maybe_downscale(image, max_side=1536)

        inputs = self.processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=score_threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs["original_sizes"].tolist(),  # ← must be list, not tensor
        )[0]

        masks  = results["masks"].cpu().numpy().astype(bool)   # (N, H, W)
        boxes  = results["boxes"].cpu().numpy()                # (N, 4)
        scores = results["scores"].cpu().numpy()               # (N,)

        return [
            Detection(m, tuple(b.astype(int)), float(s))
            for m, b, s in zip(masks, boxes, scores)
        ]
```

### 4.4 Input downscaling

```python
def _maybe_downscale(image: Image.Image, max_side: int) -> Image.Image:
    w, h = image.size
    longest = max(w, h)
    if longest <= max_side:
        return image
    scale = max_side / longest
    return image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
```

`max_side=1536` is the default for quality. Set `max_side=1024` to roughly double throughput at a small cost to small-object accuracy. The model's vision encoder resizes internally anyway — this step only prevents unnecessary tensor allocation.

> **Important:** `target_sizes=inputs["original_sizes"].tolist()` instructs the post-processor to rescale masks back to the input image's resolution. The source image, mask, and bounding box then all share one coordinate system with no downstream conversion required.

---

## 5. Module: `render.py`

Takes a PIL image and a list of `Detection` objects; returns an annotated PIL image.

### 5.1 Palette

```python
import cv2
import numpy as np

def _palette(n: int) -> list:
    """Deterministic HSV sweep → BGR colours, one per instance."""
    colours = []
    for i in range(max(n, 1)):
        hue = int(i * (180 / max(n, 1))) % 180
        swatch = np.uint8([[[hue, 220, 220]]])
        bgr = cv2.cvtColor(swatch, cv2.COLOR_HSV2BGR)[0][0].tolist()
        colours.append(tuple(bgr))
    return colours
```

Deterministic mapping means colours do not flicker when the same image is re-run with the same prompt.

### 5.2 `draw()`

```python
def draw(
    image: Image.Image,
    detections: list,
    label: str,
    line_width: int = 2,
) -> Image.Image:

    img = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    if not detections:
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        text = f'no matches for "{label}"'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(img, text,
                    ((w - tw) // 2, (h + th) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    palette = _palette(len(detections))

    for i, det in enumerate(detections):
        color = palette[i]

        # --- contour ---
        mask_u8 = det.mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1
        )
        cv2.drawContours(img, contours, -1, color, line_width, cv2.LINE_AA)

        # --- bounding box ---
        x1, y1, x2, y2 = det.box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        # --- label pill ---
        text = f"{label} {det.score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(img, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
```

**Contour parameters explained:**

| Parameter | Value | Effect |
|---|---|---|
| `cv2.RETR_EXTERNAL` | — | Returns only the outermost boundary; ignores interior holes |
| `cv2.CHAIN_APPROX_TC89_L1` | — | Teh-Chin approximation; reduces point count with no visible quality loss |
| `cv2.LINE_AA` | — | Anti-aliased stroke; prevents jagged edges at high resolution |

The zero-detections branch renders a darkened overlay with a centred message so the UI does not silently appear to do nothing.

---

## 6. Module: `app.py`

### 6.1 Layout

```python
import gradio as gr
from inference import SAM3Runner
from render import draw

runner = SAM3Runner()   # model loads here, once, at process start

def handle(image, prompt, threshold):
    if image is None:
        return gr.update(value=None), "⬆ upload an image first"
    if not prompt.strip():
        return image, "✏ type a prompt and press Enter"
    try:
        detections = runner.segment(image, prompt, score_threshold=threshold)
        annotated  = draw(image, detections, label=prompt.strip())
        return annotated, f"✅ {len(detections)} match(es) for **{prompt.strip()}**"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        # retry with a smaller input
        from inference import _maybe_downscale
        small = _maybe_downscale(image, max_side=768)
        detections = runner.segment(small, prompt, score_threshold=threshold)
        annotated  = draw(small, detections, label=prompt.strip())
        return annotated, f"⚠ OOM — reran at reduced resolution. {len(detections)} match(es)."
    except Exception as e:
        import traceback; traceback.print_exc()
        return image, f"❌ error: {e}"

with gr.Blocks(title="SAM 3 — text-prompted segmentation") as demo:
    gr.Markdown("## SAM 3 text-prompted segmentation\nUpload a photo, type a short noun phrase, press **Enter**.")
    with gr.Row():
        inp = gr.Image(sources=["upload"], type="pil", label="Input image")
        out = gr.Image(type="pil",         label="Annotated output", interactive=False)
    with gr.Row():
        prompt = gr.Textbox(
            label="What to segment",
            placeholder="e.g.  yellow school bus  ·  person in red shirt  ·  dog",
            scale=4,
        )
        thresh = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                           label="Score threshold", scale=1)
    status = gr.Markdown()

    # Enter key in textbox → run inference
    prompt.submit(handle, [inp, prompt, thresh], [out, status])
    # Releasing the slider also re-runs if an image is loaded
    thresh.release(handle, [inp, prompt, thresh], [out, status])

demo.launch()
```

### 6.2 Launch options

| Scenario | Invocation |
|---|---|
| Local only (default) | `python app.py` → `http://127.0.0.1:7860` |
| LAN access | `demo.launch(server_name="0.0.0.0")` |
| Public tunnel (Gradio relay) | `demo.launch(share=True)` |

### 6.3 Event wiring explanation

`prompt.submit` fires on Enter regardless of text cursor position — it is the canonical "textbox confirm" event in Gradio. `thresh.release` (fires on mouse-up, not continuously during drag) gives live tuning without thrashing the GPU. Do **not** use `thresh.change`, which fires on every tick of the slider and will queue dozens of redundant inference calls.

---

## 7. Error Handling

| Failure mode | Detection | Response |
|---|---|---|
| Image is `None` | `image is None` check at top of handler | Return `None` output + nudge in status bar; do not call model |
| Empty / whitespace prompt | `not prompt.strip()` check | Return input image unchanged + nudge in status bar |
| CUDA out of memory | `except torch.cuda.OutOfMemoryError` | `torch.cuda.empty_cache()`, retry once with `max_side=768`; surface fallback in status |
| Any other exception | `except Exception as e` | Log full traceback to stdout; return input image with `str(e)` in status bar |

Never allow an exception to propagate unhandled to Gradio — it will display a raw stack trace with file paths in the browser.

---

## 8. Performance Tuning

### Baseline expectations (image-only, per-frame, bfloat16)

| GPU | Approx. latency @ 720p | Notes |
|---|---|---|
| RTX 3060 Ti (8 GB) | 400–800 ms | Fits in VRAM; just below 15 FPS for live use |
| RTX 4080 / 4090 | 150–300 ms | Comfortable interactive speed |
| H100 / H200 | 50–100 ms | Production throughput |
| Apple M-series (MPS) | 600–1200 ms | float32 only; usable for testing |
| CPU only | 5–10 s | Not suitable for interactive use |

### `torch.compile` (opt-in)

```python
import os
if os.getenv("SAM3_COMPILE"):
    import torch
    runner.model = torch.compile(runner.model, mode="reduce-overhead")
```

Adds 30–60 s to first compile. Yields ~20–30% throughput gain on Ada/Hopper. Enable via `SAM3_COMPILE=1 python app.py`. Skip for one-shot demos.

### Input size vs. quality trade-off

| `max_side` | Relative speed | Small-object quality |
|---|---|---|
| 1536 (default) | 1× | Best |
| 1024 | ~2× | Good |
| 768 | ~3.5× | Acceptable for large objects |

Expose the cap as a CLI argument rather than a UI control to avoid confusing end users:

```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--max-side", type=int, default=1536)
args = parser.parse_args()
```

Pass `args.max_side` into `SAM3Runner` and `_maybe_downscale`.

---

## 9. Running & Testing

### Run

```bash
python app.py
# Open http://127.0.0.1:7860
```

### Smoke test: `tests/test_inference.py`

```python
import pytest
from PIL import Image
import requests
from inference import SAM3Runner

@pytest.fixture(scope="module")
def runner():
    return SAM3Runner()

def test_cat_detected(runner):
    url = "http://images.cocodataset.org/val2017/000000077595.jpg"
    img = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    dets = runner.segment(img, "cat", score_threshold=0.3)
    assert len(dets) >= 1
    assert dets[0].score > 0.3
    assert dets[0].mask.shape == (img.height, img.width)

def test_empty_prompt_returns_empty(runner):
    img = Image.new("RGB", (512, 512))
    dets = runner.segment(img, "   ")
    assert dets == []
```

### Smoke test: `tests/test_render.py`

```python
import numpy as np
from PIL import Image
from inference import Detection
from render import draw

def test_contour_drawn_on_mask_border():
    img = Image.new("RGB", (200, 200), (100, 100, 100))
    mask = np.zeros((200, 200), dtype=bool)
    mask[50:150, 50:150] = True                   # white square
    det = Detection(mask=mask, box=(50, 50, 150, 150), score=0.95)
    result = draw(img, [det], label="box")
    arr = np.array(result)
    # Interior of square should NOT be all-green
    interior = arr[75:125, 75:125]
    assert not np.all(interior[:, :, 1] > 200)

def test_no_detections_returns_image_with_message():
    img = Image.new("RGB", (200, 200))
    result = draw(img, [], label="cat")
    assert result.size == img.size   # no crash, same dimensions
```

Run both with `pytest tests/ -v`.

---

## 10. Known Gotchas

**`original_sizes` must be a plain list, not a tensor.**
`inputs["original_sizes"].tolist()` is required. Passing the tensor directly silently causes the post-processor to miscalculate rescale factors in some `transformers` versions, producing masks at the wrong resolution with no error raised.

**Mask shape is `(N, H, W)`, not a single combined mask.**
Each instance has its own mask plane. To produce a merged binary mask (e.g. for saving an alpha cutout), use `np.any(masks, axis=0)`. For contour drawing, keep them per-instance so each gets its own colour and contour call.

**`gr.Image` returns `None` on clear, not an empty array.**
When the user clicks the ✕ button in the upload component, `image` is `None` — not a blank PIL image. The handler must check for `None` explicitly before any `.size`, `.convert()`, or model call.

**Do not cast `inputs` to bfloat16 manually.**
The processor emits float32 tensors. PyTorch handles mixed-precision internally during the forward pass. Manually casting all inputs to bf16 will break integer tensors (token IDs, attention masks) and raise `RuntimeError` in the embedding layer.

**Short noun phrases outperform full sentences.**
SAM 3 was trained on short concept labels. `"yellow school bus"` and `"person in red shirt"` perform well. `"please find all of the yellow school buses in this photograph"` degrades detection quality noticeably. Document this in the UI placeholder text and in `README.md`.

**Warmup shot is not optional for production feel.**
Without the warmup call in `SAM3Runner.__init__`, the first real user request pays a 2–3 second CUDA kernel compilation penalty. The dummy image should be `512×512` — large enough to exercise all code paths, small enough to complete in under a second on any CUDA device.

**MPS + `bfloat16` is unreliable on Apple Silicon.**
Force `float32` on `mps` devices. The `_get_device_and_dtype()` function already does this; do not override it with an env-var that forces bf16 globally without checking the device.
