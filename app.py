# ============================================================
# NeferInterior — YOLO + SAM2 Gradio App
#
# Steps:
#   1. pip install -r requirements.txt
#   2. python verify_setup.py
#   3. Update YOLO_WEIGHTS path below
#   4. python app.py → open URL in terminal
# ============================================================

import json
import os
import numpy as np
import torch
import gradio as gr
from PIL import Image, ImageFile
from ultralytics import YOLO
from sam2 import load_model
from sam2.sam2_image_predictor import SAM2ImagePredictor

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── CONFIG ────────────────────────────────────────────────────
YOLO_WEIGHTS    = "model.pt"      # ← UPDATE: path to your model.pt
CHECKPOINT_FILE = "./checkpoints/sam2_hiera_tiny.pt"
SAM2_VARIANT    = "tiny"
DEVICE          = "cpu"
YOLO_CONF       = 0.30
GRID_SIZE       = 4

# ── COCO furniture filter ─────────────────────────────────────
COCO_TO_NEFER = {
    13: "chair",
    56: "chair",
    57: "chair",
    59: "bed",
    60: "dining_table",
}

ALL_ZONES = [
    "top-left",    "top",    "top-right",
    "mid-left",    "center", "mid-right",
    "bottom-left", "bottom", "bottom-right",
]

# SAM2 segment label colors (R,G,B) for overlay
SEG_COLORS = {
    "floor":       (0,   200, 100),
    "wall":        (80,  120, 255),
    "ceiling":     (255, 200,   0),
    "window_door": (255,  60,  60),
}


# ════════════════════════════════════════════════════════════════
# STARTUP CHECKS
# ════════════════════════════════════════════════════════════════

print("=" * 55)
print("NeferInterior — Starting")
print(f"Device    : {DEVICE}")
print(f"YOLO      : {YOLO_WEIGHTS}")
print(f"SAM2      : {CHECKPOINT_FILE}")
print("=" * 55)

if not os.path.exists(CHECKPOINT_FILE):
    raise FileNotFoundError(
        f"\nSAM2 checkpoint not found: {CHECKPOINT_FILE}\n"
        "Run: python verify_setup.py"
    )
if not os.path.exists(YOLO_WEIGHTS):
    raise FileNotFoundError(
        f"\nYOLO weights not found: {YOLO_WEIGHTS}\n"
        "Update YOLO_WEIGHTS in app.py"
    )


# ════════════════════════════════════════════════════════════════
# LOAD MODELS
# ════════════════════════════════════════════════════════════════

print("\nLoading YOLO...")
yolo_model = YOLO(YOLO_WEIGHTS)
print("YOLO OK")

print("Loading SAM2...")
sam2_model = load_model(
    variant   = SAM2_VARIANT,
    ckpt_path = CHECKPOINT_FILE,
    device    = DEVICE,
)
sam2_predictor = SAM2ImagePredictor(sam2_model)
print("SAM2 OK")
print("\nStarting — URL will appear below...\n")


# ════════════════════════════════════════════════════════════════
# YOLO — Furniture Detection + built-in visualization
# UNCHANGED — results.plot() kept exactly as original
# Only change: also returns raw `results` for combined view reuse
# ════════════════════════════════════════════════════════════════

def run_yolo(image_np: np.ndarray) -> tuple:
    img_h, img_w = image_np.shape[:2]
    results      = yolo_model(image_np, conf=YOLO_CONF, verbose=False)[0]
    furniture    = []

    if results.boxes is not None:
        for box in results.boxes:
            cls_id = int(box.cls.item())
            conf_s = float(box.conf.item())

            if cls_id not in COCO_TO_NEFER:
                continue

            label           = COCO_TO_NEFER[cls_id]
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            cx_norm = ((x1 + x2) / 2) / img_w
            cy_norm = ((y1 + y2) / 2) / img_h
            rel_w   = (x2 - x1) / img_w
            rel_h   = (y2 - y1) / img_h

            col      = "left"   if cx_norm < 0.33 else ("right"  if cx_norm > 0.66 else "center")
            row      = "top"    if cy_norm < 0.33 else ("bottom" if cy_norm > 0.66 else "mid")
            position = f"{row}-{col}" if col != "center" else row

            area = rel_w * rel_h
            size = "large" if area > 0.25 else ("medium" if area > 0.08 else "small")

            furniture.append({
                "label":      label,
                "confidence": round(conf_s, 3),
                "position":   position,
                "size":       size,
                "bbox_norm": {
                    "x1": round(x1/img_w, 3), "y1": round(y1/img_h, 3),
                    "x2": round(x2/img_w, 3), "y2": round(y2/img_h, 3),
                },
            })

    room_status = (
        "furnished"           if len(furniture) >= 2 else
        "partially_furnished" if len(furniture) == 1 else
        "empty"
    )
    occupied = list({f["position"] for f in furniture})

    analysis = {
        "room_status":     room_status,
        "furniture_count": len(furniture),
        "furniture":       furniture,
        "occupied_zones":  occupied,
        "free_zones":      [z for z in ALL_ZONES if z not in occupied],
    }

    # YOLO built-in visualization — unchanged
    yolo_vis = results.plot()           # BGR
    yolo_vis = yolo_vis[:, :, ::-1]    # BGR → RGB

    # also return raw results so combined can reuse — no second inference needed
    return analysis, yolo_vis, results


# ════════════════════════════════════════════════════════════════
# SAM2 — Room Segmentation + visualization
# UNCHANGED from original
# ════════════════════════════════════════════════════════════════

def classify_segment(cy, area, aspect, height) -> str:
    if cy > 0.55  and area > 0.08 and aspect > 1.2:    return "floor"
    if cy < 0.25  and area > 0.05 and aspect > 1.5:    return "ceiling"
    if area > 0.12 and height > 0.35 and aspect < 3.0: return "wall"
    if 0.02 < area < 0.35 and aspect < 2.5 and cy < 0.75: return "window_door"
    return "unknown"


def run_sam2(image_np: np.ndarray) -> tuple:
    img_h, img_w = image_np.shape[:2]

    xs = np.linspace(0.1, 0.9, GRID_SIZE)
    ys = np.linspace(0.1, 0.9, GRID_SIZE)
    grid_points  = np.array([
        [int(x * img_w), int(y * img_h)]
        for y in ys for x in xs
    ])
    point_labels = np.ones(len(grid_points), dtype=np.int32)

    with torch.inference_mode():
        sam2_predictor.set_image(image_np)
        masks_out, scores_out, _ = sam2_predictor.predict(
            point_coords     = grid_points,
            point_labels     = point_labels,
            multimask_output = True,
        )

    structure  = {k: [] for k in ["floor", "wall", "ceiling", "window_door", "unknown"]}
    seg_masks  = []
    seen_areas = set()

    for mask in masks_out:
        mask     = mask.astype(bool)
        area_key = round(float(mask.sum()) / (img_h * img_w), 3)
        if area_key in seen_areas:
            continue
        seen_areas.add(area_key)

        ys_m, xs_m = np.where(mask)
        if len(ys_m) == 0:
            continue

        x1, y1 = int(xs_m.min()), int(ys_m.min())
        x2, y2 = int(xs_m.max()), int(ys_m.max())
        cx     = ((x1 + x2) / 2) / img_w
        cy     = ((y1 + y2) / 2) / img_h
        area   = float(mask.sum()) / (img_h * img_w)
        width  = (x2 - x1) / img_w
        height = (y2 - y1) / img_h
        aspect = width / height if height > 0 else 0

        label = classify_segment(cy, area, aspect, height)
        props = {
            "bbox_norm": {
                "x1": round(x1/img_w, 3), "y1": round(y1/img_h, 3),
                "x2": round(x2/img_w, 3), "y2": round(y2/img_h, 3),
            },
            "center":     {"cx": round(cx, 3), "cy": round(cy, 3)},
            "area_ratio": round(area, 4),
            "aspect":     round(aspect, 3),
            "label":      label,
        }
        structure[label].append(props)
        seg_masks.append((mask, label))

    vis_img = Image.fromarray(image_np).convert("RGBA")
    for mask, label in seg_masks:
        if label not in SEG_COLORS:
            continue
        r, g, b   = SEG_COLORS[label]
        color_img = Image.new("RGBA", vis_img.size, (r, g, b, 130))
        mask_img  = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        vis_img.paste(color_img, mask=mask_img)

    sam2_vis = np.array(vis_img.convert("RGB"))

    floor_polygon  = None
    placeable_area = 0.0
    if structure["floor"]:
        best = max(structure["floor"], key=lambda x: x["area_ratio"])
        placeable_area = best["area_ratio"]
        bb = best["bbox_norm"]
        floor_polygon = [
            [bb["x1"], bb["y1"]], [bb["x2"], bb["y1"]],
            [bb["x2"], bb["y2"]], [bb["x1"], bb["y2"]],
        ]

    result = {
        "floor":          structure["floor"],
        "walls":          structure["wall"],
        "ceiling":        structure["ceiling"],
        "windows_doors":  structure["window_door"],
        "floor_polygon":  floor_polygon,
        "placeable_area": round(placeable_area, 4),
        "segment_counts": {
            "floor":         len(structure["floor"]),
            "walls":         len(structure["wall"]),
            "ceiling":       len(structure["ceiling"]),
            "windows_doors": len(structure["window_door"]),
        },
    }
    return result, sam2_vis, seg_masks


# ════════════════════════════════════════════════════════════════
# MERGE → unified dict for Groq — UNCHANGED
# ════════════════════════════════════════════════════════════════

def merge(yolo: dict, sam2: dict) -> dict:
    conflicts = []
    for f in yolo.get("furniture", []):
        fb   = f["bbox_norm"]
        f_cx = (fb["x1"] + fb["x2"]) / 2
        f_cy = (fb["y1"] + fb["y2"]) / 2
        for wd in sam2.get("windows_doors", []):
            wb = wd["bbox_norm"]
            if wb["x1"] < f_cx < wb["x2"] and wb["y1"] < f_cy < wb["y2"]:
                conflicts.append({
                    "furniture": f["label"],
                    "position":  f["position"],
                    "issue":     "overlaps window/door",
                })
    return {
        "room_status":     yolo["room_status"],
        "furniture_count": yolo["furniture_count"],
        "furniture":       yolo["furniture"],
        "occupied_zones":  yolo["occupied_zones"],
        "free_zones":      yolo["free_zones"],
        "room_structure": {
            "floor_detected":      len(sam2["floor"]) > 0,
            "walls_detected":      len(sam2["walls"]) > 0,
            "windows_doors_count": sam2["segment_counts"]["windows_doors"],
            "windows_doors":       sam2["windows_doors"],
            "floor_polygon":       sam2["floor_polygon"],
            "placeable_area_pct":  round(sam2["placeable_area"] * 100, 1),
        },
        "placement_conflicts": conflicts,
        "has_conflicts":       len(conflicts) > 0,
    }


# ════════════════════════════════════════════════════════════════
# GRADIO PIPELINE
# ════════════════════════════════════════════════════════════════

def run_pipeline(image: np.ndarray):
    if image is None:
        empty = np.zeros((400, 600, 3), dtype=np.uint8)
        return empty, empty, empty, "Please upload an image."

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if len(image.shape) == 2:
        import cv2
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        import cv2
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # ── YOLO — runs ONCE on original image ───────────────────
    print("Running YOLO...")
    yolo_result, yolo_vis, yolo_raw = run_yolo(image)
    print(f"  furniture={yolo_result['furniture_count']}  status={yolo_result['room_status']}")

    # ── SAM2 — runs on original image ────────────────────────
    print("Running SAM2 (~15-25s on CPU)...")
    sam2_result, sam2_vis, seg_masks = run_sam2(image)
    c = sam2_result["segment_counts"]
    print(f"  floor={c['floor']}  walls={c['walls']}  windows_doors={c['windows_doors']}")

    # ── Combined: reuse yolo_raw.plot() on SAM2 background ───
    # FIX: yolo_raw already has boxes from original image inference
    # just render them onto sam2_vis — NO second YOLO inference
    combined_vis = yolo_raw.plot(img=sam2_vis[:, :, ::-1])  # plot() expects BGR
    combined_vis = combined_vis[:, :, ::-1]                 # BGR → RGB

    # ── Merge ─────────────────────────────────────────────────
    unified = merge(yolo_result, sam2_result)

    output = {
        "room_status":     unified["room_status"],
        "furniture_count": unified["furniture_count"],
        "furniture":       unified["furniture"],
        "occupied_zones":  unified["occupied_zones"],
        "free_zones":      unified["free_zones"],
        "room_structure": {
            "floor_detected":      unified["room_structure"]["floor_detected"],
            "walls_detected":      unified["room_structure"]["walls_detected"],
            "windows_doors_count": unified["room_structure"]["windows_doors_count"],
            "placeable_area_pct":  unified["room_structure"]["placeable_area_pct"],
        },
        "placement_conflicts": unified["placement_conflicts"],
        "has_conflicts":       unified["has_conflicts"],
    }

    return yolo_vis, sam2_vis, combined_vis, json.dumps(output, indent=2)


# ════════════════════════════════════════════════════════════════
# GRADIO UI — UNCHANGED from original
# ════════════════════════════════════════════════════════════════

with gr.Blocks(title="NeferInterior — Room Analysis", theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    # 🏠 NeferInterior — Room Analysis
    **YOLO** detects furniture &nbsp;|&nbsp;
    **SAM2** segments room structure &nbsp;|&nbsp;
    JSON feeds into **Groq** (next stage)

    > ⏱️ SAM2 takes ~20-30s on CPU — please wait after clicking Analyze
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Room Photo", type="numpy", height=360)
            run_btn     = gr.Button("🔍 Analyze Room", variant="primary", size="lg")
            gr.Markdown("""
            **YOLO detects (furniture only):**
            Chair / Couch / Bench → `chair`
            Bed → `bed`
            Dining Table → `dining_table`

            **SAM2 segments:**
            🟩 Floor &nbsp; 🟦 Wall &nbsp; 🟨 Ceiling &nbsp; 🟥 Window/Door
            """)

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("🟦 YOLO — Furniture"):
                    yolo_out = gr.Image(label="YOLO built-in visualization", height=360)
                with gr.Tab("🟩 SAM2 — Room Structure"):
                    sam2_out = gr.Image(label="SAM2 mask overlay", height=360)
                with gr.Tab("🔀 Combined"):
                    combined_out = gr.Image(label="SAM2 + YOLO combined", height=360)
                with gr.Tab("📊 JSON → Groq"):
                    json_out = gr.Code(
                        label    = "Unified Analysis — feeds into Groq next stage",
                        language = "json",
                        lines    = 28,
                    )

    run_btn.click(
        fn      = run_pipeline,
        inputs  = [input_image],
        outputs = [yolo_out, sam2_out, combined_out, json_out],
    )

    gr.Markdown("""
    ---
    **Pipeline:** YOLO → SAM2 → Groq Layout Engine → Stable Diffusion + ControlNet → Render
    """)


if __name__ == "__main__":
    demo.launch(
        server_name = "127.0.0.1",
        server_port = 7860,
        show_error  = True,
        share       = False,
    )