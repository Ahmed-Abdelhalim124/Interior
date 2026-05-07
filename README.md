# NeferInterior — AI-Powered Room Redesign System

NeferInterior is an end-to-end interior design pipeline that takes a room photograph as input and produces a photorealistic redesign in any chosen architectural style. The system combines computer vision, large language models, and generative image synthesis into a single automated workflow.



# DEMO LINK : https://drive.google.com/file/d/1d7ogCRPntsNHICa-GYsfxFmrK23B65x_/view?usp=sharing




## Overview

NeferInterior automates the interior design process by:

1. **Detecting** all furniture items in the room using a custom-trained object detection model
2. **Understanding** the room structure — identifying floor, walls, ceiling, doors, and windows with pixel-level accuracy
3. **Analyzing** the layout intelligently using a large language model that reviews both the visual and spatial data
4. **Generating** a photorealistic redesigned version of the room guided by the structural depth map

The result is delivered in approximately **17 seconds** on GPU hardware, with zero manual configuration required from the user.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     INPUT: Room Photo                    │
└─────────────────────┬───────────────────────────────────┘
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
  ┌───────────────┐     ┌──────────────────┐
  │  YOLO v8      │     │  SegFormer-b5    │
  │  Furniture    │     │  Room Structure  │
  │  Detection    │     │  Segmentation    │
  └───────┬───────┘     └────────┬─────────┘
          │                      │
          │   ┌──────────────────┘
          ▼   ▼
  ┌───────────────┐
  │  Merge Layer  │  Conflict detection, free zone mapping
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐
  │  Groq LLM     │  Layout analysis + SD prompt generation
  │  Llama-4      │
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐
  │  MiDaS        │  Depth map extraction
  └───────┬───────┘
          │
          ▼
  ┌───────────────┐
  │  SD 1.5 +     │  Photorealistic image generation
  │  ControlNet   │
  └───────┬───────┘
          │
┌─────────┴───────────────────────────────────────────────┐
│                  OUTPUT: Redesigned Room                  │
└─────────────────────────────────────────────────────────┘
```

### Stage Descriptions

**Stage 1A — Furniture Detection (YOLO)**  (model link: https://drive.google.com/file/d/1Js2VCHm8ZySYtfI7c4BHdfEf1Wx7S3lw/view?usp=sharing )
A custom-trained YOLOv8 model detects furniture items including chairs, beds, and dining tables. Each detected item is assigned a normalized bounding box, confidence score, spatial zone (e.g. mid-left, top-right), and size classification (small / medium / large).

**Stage 1B — Room Segmentation (SegFormer)**  
SegFormer-b5 fine-tuned on ADE20K provides pixel-level semantic segmentation across 150 classes. The system extracts floor, wall, ceiling, window, and door regions with precise boundaries. A 3×3 spatial grid is applied to the floor mask to identify free zones available for furniture placement.

**Stage 1C — Merge & Conflict Detection**  
YOLO and SegFormer outputs are combined into a unified room analysis JSON. The merge layer detects placement conflicts — for example, furniture positioned in front of a door or window.

**Stage 2 — Layout Intelligence (Groq)**  
The unified JSON and the original room image are sent to Groq's Llama-4-Scout model. The LLM cross-checks the automated analysis against the image, corrects any detection errors, suggests optimal furniture rearrangement, and generates a detailed Stable Diffusion prompt tailored to the chosen style.

**Stage 3A — Depth Estimation (MiDaS)**  
Intel's DPT-hybrid-MiDaS model generates a depth map from the room image. This map preserves the 3D spatial structure of the room and guides the image generation step.

**Stage 3B — Image Generation (SD 1.5 + ControlNet)**  
Stable Diffusion 1.5 conditioned on the ControlNet depth map generates the final redesigned room. The Groq-generated prompt ensures the output reflects the selected style while maintaining the original room's spatial layout.

---

## Models & Technologies

| Component | Model | Version | Purpose |
|---|---|---|---|
| Furniture Detection | Custom YOLOv8 | `model.pt` | Detects chairs, beds, tables |
| Room Segmentation | SegFormer-b5-ADE20K | `nvidia/segformer-b5-finetuned-ade-640-640` | Floor, wall, ceiling, door, window |
| Layout Intelligence | Llama-4-Scout | `meta-llama/llama-4-scout-17b-16e-instruct` | Layout analysis + prompt generation |
| Depth Estimation | MiDaS DPT-Hybrid | `Intel/dpt-hybrid-midas` | Structural depth map |
| Image Generation | Stable Diffusion 1.5 | `runwayml/stable-diffusion-v1-5` | Room redesign |
| Generation Control | ControlNet Depth | `lllyasviel/sd-controlnet-depth` | Structural guidance |
| Generation Control | ControlNet Scribble | `lllyasviel/sd-controlnet-scribble` | Edge-based guidance |
| LLM API | Groq Cloud | — | Fast inference for Llama-4 |

---

## System Requirements

### Kaggle (Recommended)
| Requirement | Specification |
|---|---|
| Accelerator | GPU P100 (16GB VRAM) |
| Internet | Enabled |
| Disk | 20GB available |


---

## Deployment Guide (Kaggle)

### Step 1 — Notebook Settings
- Set **Accelerator** to `GPU P100`
- Enable **Internet**

### Step 2 — Cell 1: Install Dependencies
```python
!pip install -q gradio groq controlnet-aux diffusers transformers accelerate ultralytics
```

### Step 3 — Cell 2: Setup Models
```python
import shutil, os
from huggingface_hub import snapshot_download

os.makedirs("./models/controlnet/depth",    exist_ok=True)
os.makedirs("./models/controlnet/scribble", exist_ok=True)

shutil.copy(
    "/kaggle/input/datasets/ahmedabdelhalim2/neferinterior/model.pt",
    "./model.pt"
)
print("model.pt:", os.path.exists("./model.pt"))

snapshot_download("lllyasviel/sd-controlnet-depth",    local_dir="./models/controlnet/depth")
snapshot_download("lllyasviel/sd-controlnet-scribble", local_dir="./models/controlnet/scribble")
print("All ready ✅")
```

> SegFormer, MiDaS, and SD 1.5 are downloaded automatically from HuggingFace on the first pipeline run. No manual download is required for these models.

### Step 4 — Cell 3: Launch Application
Paste the full contents of `app.py` into Cell 3 and run.

On successful startup the terminal will display:
```
=======================================================
NeferInterior — GPU Pipeline
  YOLO      : ./model.pt
  SegFormer : nvidia/segformer-b5-finetuned-ade-640-640
  MiDaS     : Intel/dpt-hybrid-midas
  SD 1.5    : runwayml/stable-diffusion-v1-5
  CUDA      : True
  GPU       : Tesla P100-PCIE-16GB
  Device    : cuda
=======================================================
Loading YOLO...
YOLO OK
* Running on public URL: https://xxxxxxxx.gradio.live
```

Open the public URL in any browser to access the application.

---

## User Interface

The interface is organized into two sections:

### Main View
| Element | Description |
|---|---|
| Room Photo Upload | Accepts any JPEG/PNG room photograph |
| Style Selector | Dropdown with 8 interior design styles |
| Analyze & Redesign Button | Triggers the full pipeline in one click |
| Status Bar | Live progress updates for each pipeline stage |
| Redesigned Room | Full-size output image displayed prominently |

### Analysis Details (Expandable)
| Tab | Content |
|---|---|
| 🟦 YOLO — Furniture | Bounding box visualization of detected furniture |
| 🟩 SegFormer — Room Structure | Color-coded semantic segmentation overlay |
| 🔀 Combined | YOLO detections overlaid on SegFormer segmentation |
| 🧠 Groq Layout Analysis | Full LLM output: corrections, layout suggestions, SD prompt |


---

## Output Examples

### Segmentation Color Map

The SegFormer tab uses the following color coding to visualize room structure:

| Color | Segment |
|---|---|
| 🟩 Green | Floor |
| 🟦 Blue | Wall |
| 🟨 Yellow | Ceiling |
| 🔴 Red | Window |
| 🟠 Orange | Door |

### Groq Analysis Output

The Groq tab displays:
- **Room Assessment** — detected style, lighting conditions, and any corrections to YOLO/SegFormer outputs
- **Layout Suggestions** — recommended furniture repositioning with reasoning
- **Best Zone for New Furniture** — identified free zones based on floor area analysis
- **Stable Diffusion Prompt** — the exact prompt used to generate the redesign
- **ControlNet Recommendation** — depth or scribble mode with justification

---

## Project Structure

```
NeferInterior/
├── app.py                        
├── requirements.txt              
├── README.md                     
├── model.pt                      
└── models/
    └── controlnet/
        ├── depth/                
        └── scribble/             
```


---
