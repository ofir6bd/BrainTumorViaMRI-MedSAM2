"""Builds a 4-slide PowerPoint presentation summarizing the YOLO tumour-segmentation
pipeline, using real numbers from the current best checkpoint's sidecar JSON and the
real rendered assets in `_slide_assets/` (see `_gen_slide_assets.py`).

Run once:
    .venv\\Scripts\\python.exe YOLO\\_build_presentation.py
"""
import json
import os

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

_HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(_HERE, "_slide_assets")
META_PATH = os.path.join(_HERE, "weights", "best_20260821-013417_valloss2p9192_testdice0p8662.json")
OUT_PATH = os.path.join(_HERE, "YOLO_Pipeline_Presentation.pptx")

with open(META_PATH, "r", encoding="utf-8") as f:
    META = json.load(f)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
NAVY = RGBColor(0x14, 0x14, 0x28)
BLUE = RGBColor(0x3B, 0x82, 0xF6)
GREEN = RGBColor(0x10, 0xB9, 0x81)
RED = RGBColor(0xEF, 0x44, 0x44)
GOLD = RGBColor(0xF5, 0xA6, 0x23)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREY = RGBColor(0x6B, 0x72, 0x80)
LIGHT_BG = RGBColor(0xF7, 0xF8, 0xFC)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width = SLIDE_W
prs.slide_height = SLIDE_H
BLANK = prs.slide_layouts[6]


def add_slide():
    return prs.slides.add_slide(BLANK)


def fill_bg(slide, color=LIGHT_BG):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = color
    bg.line.fill.background()
    bg.shadow.inherit = False
    # send to back
    spTree = slide.shapes._spTree
    spTree.remove(bg._element)
    spTree.insert(2, bg._element)
    return bg


def header_band(slide, title, subtitle=None, band_color=NAVY):
    band = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, Inches(1.15))
    band.fill.solid()
    band.fill.fore_color.rgb = band_color
    band.line.fill.background()
    band.shadow.inherit = False

    tb = slide.shapes.add_textbox(Inches(0.5), Inches(0.12), Inches(10.5), Inches(0.6))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = WHITE

    if subtitle:
        tb2 = slide.shapes.add_textbox(Inches(0.5), Inches(0.68), Inches(11.5), Inches(0.4))
        tf2 = tb2.text_frame
        p2 = tf2.paragraphs[0]
        p2.text = subtitle
        p2.font.size = Pt(14)
        p2.font.color.rgb = RGBColor(0xC9, 0xCE, 0xF2)


def add_bullets(slide, left, top, width, height, items, size=16, color=RGBColor(0x22, 0x27, 0x3A),
                 bold_first=False, line_spacing=1.15):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if isinstance(item, tuple):
            text, kwargs = item
        else:
            text, kwargs = item, {}
        p.text = text
        p.font.size = Pt(kwargs.get("size", size))
        p.font.bold = kwargs.get("bold", bold_first and i == 0)
        p.font.color.rgb = kwargs.get("color", color)
        p.space_after = Pt(kwargs.get("space_after", 8))
        p.line_spacing = line_spacing
        p.level = kwargs.get("level", 0)
    return tb


def stat_card(slide, left, top, width, height, value, label, value_color=BLUE):
    card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    card.fill.solid()
    card.fill.fore_color.rgb = WHITE
    card.line.color.rgb = RGBColor(0xE2, 0xE5, 0xF0)
    card.line.width = Pt(1)
    card.shadow.inherit = False
    tf = card.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p1 = tf.paragraphs[0]
    p1.text = value
    p1.font.size = Pt(28)
    p1.font.bold = True
    p1.font.color.rgb = value_color
    p1.alignment = PP_ALIGN.CENTER
    p2 = tf.add_paragraph()
    p2.text = label
    p2.font.size = Pt(12)
    p2.font.color.rgb = GREY
    p2.alignment = PP_ALIGN.CENTER
    return card


def picture_fit(slide, path, left, top, max_w, max_h):
    from PIL import Image
    with Image.open(path) as im:
        w_px, h_px = im.size
    ratio = w_px / h_px
    w, h = max_w, max_w / ratio
    if h > max_h:
        h = max_h
        w = max_h * ratio
    x = left + (max_w - w) / 2
    y = top + (max_h - h) / 2
    slide.shapes.add_picture(path, x, y, width=Emu(int(w)), height=Emu(int(h)))


# ===========================================================================
# Slide 1 — Pipeline overview
# ===========================================================================
s1 = add_slide()
fill_bg(s1)
header_band(s1, "YOLO Tumour Segmentation — Pipeline Overview",
            "BraTS whole-tumour segmentation on axial MRI slices, fine-tuned YOLO11n-seg")

steps = [
    ("1. Patient split", "1,621 labeled patients -> train 60% / val 20% / test 20%, by patient (seed=42)"),
    ("2. RGB frame per slice", "R = T1C-T1 (uptake)   G = T2   B = FLAIR  ->  512x512 image"),
    ("3. Labels from expert mask", "seg>0 -> connected components -> polygons -> YOLO-seg .txt"),
    ("4. Train yolo11n-seg", "Ultralytics fine-tune on train/val, 25 epochs, imgsz 512"),
    ("5. Evaluate on held-out test", "Dice = 2|pred (cap) gt| / (|pred|+|gt|)  ->  mean Dice per patient"),
    ("6. Web UI", "Flask /yolo blueprint renders GT vs. prediction overlay + Dice tables"),
]

card_w = Inches(3.85)
card_h = Inches(1.75)
gap_x = Inches(0.25)
gap_y = Inches(0.25)
start_x = Inches(0.5)
start_y = Inches(1.55)
colors = [BLUE, BLUE, GOLD, GREEN, GREEN, BLUE]
for i, (title, desc) in enumerate(steps):
    col = i % 3
    row = i // 3
    x = start_x + col * (card_w + gap_x)
    y = start_y + row * (card_h + gap_y)
    card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, card_w, card_h)
    card.fill.solid()
    card.fill.fore_color.rgb = WHITE
    card.line.color.rgb = RGBColor(0xE2, 0xE5, 0xF0)
    card.line.width = Pt(1)
    card.shadow.inherit = False
    accent = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, Inches(0.08), card_h)
    accent.fill.solid()
    accent.fill.fore_color.rgb = colors[i]
    accent.line.fill.background()
    accent.shadow.inherit = False
    tf = card.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.25)
    tf.margin_right = Inches(0.2)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p1 = tf.paragraphs[0]
    p1.text = title
    p1.font.size = Pt(16)
    p1.font.bold = True
    p1.font.color.rgb = NAVY
    p2 = tf.add_paragraph()
    p2.text = desc
    p2.font.size = Pt(11.5)
    p2.font.color.rgb = RGBColor(0x44, 0x49, 0x5C)
    p2.space_before = Pt(4)

foot = s1.shapes.add_textbox(Inches(0.5), Inches(6.95), Inches(12), Inches(0.4))
p = foot.text_frame.paragraphs[0]
p.text = "Code: YOLO/pipeline.py, train.py, evaluate.py, render.py, routes.py"
p.font.size = Pt(11)
p.font.italic = True
p.font.color.rgb = GREY

# ===========================================================================
# Slide 2 — Dataset & patient split
# ===========================================================================
s2 = add_slide()
fill_bg(s2)
header_band(s2, "Dataset & Patient Split",
            "All patients are labeled (expert -seg mask) — split by patient, never by slice")

stat_card(s2, Inches(0.6), Inches(1.6), Inches(2.9), Inches(1.5), "1,621", "Total labeled patients", BLUE)
stat_card(s2, Inches(3.7), Inches(1.6), Inches(2.9), Inches(1.5), "973", "Train (60%)", GREEN)
stat_card(s2, Inches(6.8), Inches(1.6), Inches(2.9), Inches(1.5), "324", "Validation (20%)", GOLD)
stat_card(s2, Inches(9.9), Inches(1.6), Inches(2.85), Inches(1.5), "324", "Test (20%, held out)", RED)

add_bullets(s2, Inches(0.6), Inches(3.4), Inches(12.1), Inches(3.2), [
    ("Source pool (config.yaml -> paths.extract_to):", {"bold": True, "size": 17, "color": NAVY, "space_after": 10}),
    ("  • training_data_additional — 1,621 labeled patients", {"size": 15}),
    ("  • pre-split on disk into train/ , val/ and test/ subfolders", {"size": 15, "space_after": 18}),
    ("Split logic (pipeline.split_patients):", {"bold": True, "size": 17, "color": NAVY, "space_after": 10}),
    ("  • Read straight from the folders — no reshuffling, stable by construction", {"size": 15}),
    ("  • Split by patient (all slices of one patient stay in one split) — avoids data leakage", {"size": 15}),
    ("  • Train + val used for fine-tuning yolo11n-seg; test is never trained on", {"size": 15}),
    ("  • Test split = the web UI's patient list; every Dice shown there is on unseen patients", {"size": 15}),
], size=15)

# ===========================================================================
# Slide 3 — Training results
# ===========================================================================
s3 = add_slide()
fill_bg(s3)
header_band(s3, "Training Results — Current Best Checkpoint",
            os.path.basename(META_PATH).replace(".json", ".pt"))

picture_fit(s3, os.path.join(ASSETS, "training_curves.png"),
            Inches(0.5), Inches(1.4), Inches(8.0), Inches(4.6))

n_test = META["n_test_patients"]
stat_card(s3, Inches(8.75), Inches(1.4), Inches(4.0), Inches(1.15), f"{META['test_dice']:.3f}", "Mean test Dice", GREEN)
stat_card(s3, Inches(8.75), Inches(2.7), Inches(4.0), Inches(1.15), f"{META['val_loss']:.3f}", "Best val loss (total)", BLUE)
stat_card(s3, Inches(8.75), Inches(4.0), Inches(4.0), Inches(1.15), f"{n_test}", "Test patients evaluated", GOLD)

add_bullets(s3, Inches(8.75), Inches(5.3), Inches(4.0), Inches(1.6), [
    (f"Epochs: {META['epochs']}", {"size": 13}),
    (f"Image size: {META['imgsz']}", {"size": 13}),
    (f"Batch size: {META['batch']}", {"size": 13}),
    ("Base model: yolo11n-seg.pt (COCO pretrained)", {"size": 13}),
], size=13)

foot3 = s3.shapes.add_textbox(Inches(0.5), Inches(6.95), Inches(8), Inches(0.4))
p = foot3.text_frame.paragraphs[0]
p.text = "Loss = box + seg + cls + dfl (Ultralytics results.csv). mAP@50 computed on the validation split."
p.font.size = Pt(11)
p.font.italic = True
p.font.color.rgb = GREY

# ===========================================================================
# Slide 4 — Best.pt segmentation result
# ===========================================================================
s4 = add_slide()
fill_bg(s4)
header_band(s4, "Segmentation Result — Ground Truth vs. Prediction",
            "Rendered by render.py using the checkpoint above, on a held-out test-split patient")

picture_fit(s4, os.path.join(ASSETS, "segmentation_result.png"),
            Inches(0.4), Inches(1.35), Inches(12.5), Inches(4.9))

# find the representative patient/dice used when generating the asset
rep_dice = None
rep_patient = None
target = META["test_dice"]
best_match = min(META["per_patient_test_dice"], key=lambda r: abs(r["mean_dice"] - target))
rep_patient = best_match["patient_id"]
rep_dice = best_match["mean_dice"]

cap = s4.shapes.add_textbox(Inches(0.5), Inches(6.35), Inches(12.3), Inches(0.9))
tf = cap.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = (f"Patient {rep_patient} (test split, unseen during training) — patient mean Dice "
          f"{rep_dice:.3f}, close to the overall test mean of {target:.3f} across {n_test} patients.")
p.font.size = Pt(13.5)
p.font.color.rgb = RGBColor(0x33, 0x38, 0x4A)
p2 = tf.add_paragraph()
p2.text = "Green = expert ground truth  •  Red = YOLO prediction  •  Yellow = overlap"
p2.font.size = Pt(12.5)
p2.font.bold = True
p2.font.color.rgb = NAVY
p2.space_before = Pt(4)

prs.save(OUT_PATH)
print(f"[pptx] saved -> {OUT_PATH}")
