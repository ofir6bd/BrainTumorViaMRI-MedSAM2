"""Builds Presentation/BrainTumour_YOLO_MedSAM2.pptx (15 slides) from the images in Presentation/images.

Needs python-pptx (pip install python-pptx). Run from the repo root:
    .venv\\Scripts\\python.exe Presentation\\make_charts.py
    .venv\\Scripts\\python.exe Presentation\\build_presentation.py
Every number on the slides comes from the saved run results (see outline.md for the sources).
"""
import os
import re

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")
OUT = os.path.join(HERE, "BrainTumour_YOLO_MedSAM2.pptx")

BG, PANEL, TEXT, MUTED = "0F1220", "1A1F33", "E8EAF2", "9AA0B8"
YOLO_C, MS_C, GOOD, LINE = "3987E5", "E06A2E", "2FB35C", "2A2F47"
FONT = "Segoe UI"
W, H = Inches(13.333), Inches(7.5)
TOTAL = 15


def rgb(h):
    return RGBColor.from_string(h)


# ---------------------------------------------------------------- helpers
def runs(par, text, size, color=TEXT, bold=False):
    """Add text to a paragraph; **bold** segments are bold."""
    for i, part in enumerate(re.split(r"\*\*", text)):
        if not part:
            continue
        r = par.add_run()
        r.text = part
        r.font.size, r.font.name = Pt(size), FONT
        r.font.bold = bold or i % 2 == 1
        r.font.color.rgb = rgb(color)


def box(slide, l, t, w, h, fill=None, line=None, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, l, t, w, h)
    if fill:
        s.fill.solid(); s.fill.fore_color.rgb = rgb(fill)
    else:
        s.fill.background()
    if line:
        s.line.color.rgb = rgb(line); s.line.width = Pt(1)
    else:
        s.line.fill.background()
    s.shadow.inherit = False
    return s


def text(slide, l, t, w, h, lines, size=16, color=TEXT, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space=6, bullet=False):
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    for i, item in enumerate(lines if isinstance(lines, list) else [lines]):
        lvl = 0
        if isinstance(item, tuple):
            item, lvl = item
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space)
        prefix = ("•  " if lvl == 0 else "–  ") if bullet else ""
        if lvl:
            p.level = 1
        runs(p, prefix + item, size - 2 * lvl, color if not lvl else MUTED)
    return tb


def picture(slide, name, l, t, max_w, max_h, caption=None, center=True):
    path = os.path.join(IMG, name)
    iw, ih = Image.open(path).size
    scale = min(max_w / iw, max_h / ih)
    w, h = int(iw * scale), int(ih * scale)
    x = l + (max_w - w) // 2 if center else l
    y = t + (max_h - h) // 2 if center else t
    pic = slide.shapes.add_picture(path, x, y, w, h)
    pic.line.color.rgb = rgb(LINE); pic.line.width = Pt(0.75)
    if caption:
        text(slide, x, y + h + Inches(0.04), w, Inches(0.35), caption, size=11, color=MUTED, align=PP_ALIGN.CENTER)
    return pic


def new_slide(prs, title, n, kicker=None):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid(); s.background.fill.fore_color.rgb = rgb(BG)
    if kicker:
        text(s, Inches(0.6), Inches(0.28), Inches(9), Inches(0.35), kicker.upper(), size=12, color=MS_C)
    text(s, Inches(0.6), Inches(0.52), Inches(12.1), Inches(0.8), title, size=30, color=TEXT, space=0)
    box(s, Inches(0.62), Inches(1.28), Inches(1.1), Inches(0.06), fill=MS_C)
    text(s, Inches(12.0), Inches(7.05), Inches(1.0), Inches(0.3), f"{n} / {TOTAL}", size=11, color=MUTED, align=PP_ALIGN.RIGHT)
    return s


def takeaway(slide, msg):
    l, t, w, h = Inches(0.6), Inches(6.35), Inches(12.1), Inches(0.62)
    box(slide, l, t, w, h, fill=PANEL)
    box(slide, l, t, Inches(0.08), h, fill=MS_C)
    tb = text(slide, l + Inches(0.25), t, w - Inches(0.35), h, f"**Key takeaway:** {msg}", size=16, anchor=MSO_ANCHOR.MIDDLE, space=0)
    return tb


def notes(slide, summary, key, talk=()):
    tf = slide.notes_slide.notes_text_frame
    tf.text = f"SUMMARY: {summary}\n\nKEY TAKEAWAY: {key}"
    for line in talk:
        tf.add_paragraph().text = "- " + line


# ---------------------------------------------------------------- slides
def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # 1 ---- title
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid(); s.background.fill.fore_color.rgb = rgb(BG)
    text(s, Inches(0.7), Inches(1.0), Inches(8), Inches(0.4), "CLASS PROJECT · OCTOBER 2026", size=13, color=MS_C)
    text(s, Inches(0.7), Inches(1.45), Inches(11.5), Inches(1.6), "Brain Tumour Segmentation on MRI with YOLO + MedSAM2", size=40, space=0)
    box(s, Inches(0.72), Inches(3.0), Inches(1.4), Inches(0.07), fill=MS_C)
    text(s, Inches(0.7), Inches(3.2), Inches(11.5), Inches(0.9),
         "A fast 2D detector finds the tumour; a promptable video model refines it in 3D. BraTS 2024 adult glioma, 1,621 MRI scans.",
         size=18, color=MUTED)
    picture(s, "ms_panels.png", Inches(0.7), Inches(4.25), Inches(11.9), Inches(2.4),
            caption="One slice: the MRI picture · YOLO's hint · YOLO's own mask · MedSAM2's answer (green = found, red = extra)")
    text(s, Inches(0.7), Inches(6.95), Inches(8), Inches(0.35), "Presented by: ______________", size=13, color=MUTED)
    notes(s, "Introduces the project: segmenting brain tumours on MRI by combining YOLO and MedSAM2.",
          "We combine a fast 2D detector with a 3D-aware promptable model to outline brain tumours.",
          ["The picture shows the whole pipeline on one slice, left to right."])

    # 2 ---- background: why
    s = new_slide(prs, "Why segment brain tumours automatically?", 2, "Background")
    text(s, Inches(0.6), Inches(1.6), Inches(7.2), Inches(4.6), [
        "**Gliomas** are the most common malignant primary brain tumours in adults.",
        "Doctors outline the tumour on MRI to **plan surgery and radiotherapy** and to **track change** between scans.",
        "Outlining by hand is **slow**: a scan has well over a hundred slices, each drawn separately.",
        "Experts **don't agree perfectly** with each other, so outlines vary between readers.",
        "Automation goal: a **fast, consistent 3D outline** a doctor only needs to check.",
    ], size=18, bullet=True, space=12)
    picture(s, "mri_flair_seg.png", Inches(8.3), Inches(1.55), Inches(4.4), Inches(4.5),
            caption="An expert outline on FLAIR (green = swelling, yellow = surgery cavity)")
    takeaway(s, "Automatic 3D segmentation saves expert time and makes tumour measurements consistent.")
    notes(s, "Explains the clinical need for automatic tumour outlining.",
          "Automatic 3D segmentation saves expert time and makes tumour measurements consistent.",
          ["Manual contouring is the bottleneck; variability between experts limits how 'perfect' any model can look."])

    # 3 ---- background: MRI + BraTS
    s = new_slide(prs, "MRI sequences and the BraTS 2024 data", 3, "Background")
    for i, (f, cap) in enumerate([("mri_t1c.png", "T1C (with contrast)"), ("mri_t2.png", "T2"), ("mri_flair.png", "FLAIR"),
                                  ("mri_flair_seg.png", "FLAIR + expert labels")]):
        picture(s, f, Inches(0.6 + i * 2.05), Inches(1.6), Inches(1.9), Inches(2.3), caption=cap)
    text(s, Inches(0.6), Inches(4.35), Inches(8.2), Inches(2.0), [
        "Each scan: 4 MRI sequences, 182 × 218 × 182 voxels of 1 mm³ (1,000 voxels = 1 mL).",
        "Experts label 4 parts: **NETC** dead core, **SNFH** swelling, **ET** enhancing tumour, **RC** surgery cavity.",
        "Our target: the **whole tumour** = all labelled parts together.",
    ], size=16, bullet=True, space=8)
    box(s, Inches(9.0), Inches(1.6), Inches(3.7), Inches(4.5), fill=PANEL)
    text(s, Inches(9.2), Inches(1.75), Inches(3.4), Inches(4.3), [
        "**How we score: 3D Dice**",
        "Dice = 2 × overlap ÷ (model + expert)",
        "0 = no overlap, 1 = perfect",
        "Computed per patient over the whole 3D volume, then averaged.",
        "Each sequence shows the tumour differently, so we feed three of them (T1C, T2, FLAIR) as one colour picture.",
    ], size=15, space=10)
    takeaway(s, "Different MRI sequences show different tumour parts; we score the whole-tumour outline with 3D Dice.")
    notes(s, "Shows the MRI sequences, the four labelled tumour parts and how results are scored.",
          "Different MRI sequences show different tumour parts; we score the whole-tumour outline with 3D Dice.",
          ["85% of the scans are post-operative (they contain a surgery cavity)."])

    # 4 ---- background: models
    s = new_slide(prs, "Two models with different strengths", 4, "Background")
    for i, (name, col, items) in enumerate([
        ("YOLO11-seg  (Ultralytics)", YOLO_C, [
            "Fast object detector that also draws a mask for each shape it finds.",
            "Works on **one 2D slice at a time**, no knowledge of neighbouring slices.",
            "Gives a **confidence** for every shape (0–1).",
            "We use the medium model (yolo11m-seg), pretrained on everyday photos."]),
        ("SAM2 → MedSAM2", MS_C, [
            "**Segment Anything 2** (Meta): outlines an object from a **prompt** (point, box or mask).",
            "Treats a 3D scan **like a video**: remembers the slices it already did.",
            "**MedSAM2** = SAM2 further trained on medical images and scans.",
            "39 M weights: image encoder 70%, memory 19%, mask decoder 11%, prompt encoder <0.1%."])]):
        l = Inches(0.6 + i * 6.15)
        box(s, l, Inches(1.6), Inches(5.95), Inches(4.5), fill=PANEL)
        box(s, l, Inches(1.6), Inches(5.95), Inches(0.08), fill=col)
        text(s, l + Inches(0.25), Inches(1.8), Inches(5.5), Inches(0.5), f"**{name}**", size=20, color=col)
        text(s, l + Inches(0.25), Inches(2.45), Inches(5.5), Inches(3.6), items, size=16, bullet=True, space=10)
    takeaway(s, "YOLO is good at finding the tumour on a slice; MedSAM2 can refine it using the slices around it.")
    notes(s, "Introduces the two models and why they complement each other.",
          "YOLO is good at finding the tumour on a slice; MedSAM2 can refine it using the slices around it.",
          ["Weight counts measured on the actual checkpoint: 38,962,498 in total."])

    # 5 ---- question
    s = new_slide(prs, "Research question and ground rules", 5, "The project")
    box(s, Inches(0.6), Inches(1.6), Inches(12.1), Inches(1.1), fill=PANEL)
    text(s, Inches(0.85), Inches(1.6), Inches(11.6), Inches(1.1),
         "Can MedSAM2, prompted by YOLO on every slice, outline the whole tumour **better than YOLO alone** (3D Dice)?",
         size=21, anchor=MSO_ANCHOR.MIDDLE)
    text(s, Inches(0.6), Inches(3.0), Inches(6.0), Inches(3.2), [
        "**Ground rules**",
        ("The expert mask is never shown to a model when it predicts.", 1),
        ("Settings are chosen on validation pools; test is scored once.", 1),
        ("Patients are split by person, never by scan.", 1),
        ("We compare against **YOLO's own best setup**, not a weak baseline.", 1),
    ], size=17, space=8)
    text(s, Inches(6.9), Inches(3.0), Inches(5.8), Inches(3.2), [
        "**What we built**",
        ("A fine-tuned YOLO (stage 1).", 1),
        ("A fine-tuned MedSAM2 that reads YOLO's hints (stage 2).", 1),
        ("An interactive web app to explore the data and every result.", 1),
    ], size=17, space=8)
    takeaway(s, "A fair two-stage test: does adding MedSAM2 beat the best YOLO we could build?")
    notes(s, "States the research question and the rules that keep the comparison fair.",
          "A fair two-stage test: does adding MedSAM2 beat the best YOLO we could build?")

    # 6 ---- data split
    s = new_slide(prs, "Data: 1,621 scans from 731 people, split by person", 6, "Data")
    picture(s, "an_manifest.png", Inches(0.6), Inches(1.55), Inches(5.6), Inches(4.6))
    text(s, Inches(6.5), Inches(1.6), Inches(6.2), Inches(4.6), [
        "Five pools: YOLO train 471 · YOLO val 81 · MedSAM2 train 500 · MedSAM2 val 245 · **test 324** scans.",
        "**No person is in both a YOLO pool and a MedSAM2 pool**, so MedSAM2 trains on hints YOLO made for people it never saw, like real use.",
        "Pool sizes set by learning curves: MedSAM2 barely improved past 500 patients (250 → 500 → 721: 0.884 → 0.886 → 0.886), while YOLO still gained from more data.",
        "Caveat: test shares some people (other scans) with training pools, from the original split; kept so all test scores stay comparable.",
    ], size=16, bullet=True, space=10)
    takeaway(s, "Keeping the two models' people apart makes MedSAM2's training hints realistic.")
    notes(s, "Shows how the data was split into five pools and why.",
          "Keeping the two models' people apart makes MedSAM2's training hints realistic.",
          ["221 scans were moved from MedSAM2 train to YOLO train on 2026-09-29 after the learning curves."])

    # 7 ---- analytics app
    s = new_slide(prs, "Knowing the data: an interactive analytics app", 7, "Data")
    picture(s, "an_kpis.png", Inches(0.6), Inches(1.5), Inches(12.1), Inches(1.9))
    picture(s, "an_heat.png", Inches(0.6), Inches(3.5), Inches(4.0), Inches(2.75))
    text(s, Inches(4.9), Inches(3.55), Inches(7.8), Inches(2.7), [
        "4 pages: **Analytics**, **Results**, **YOLO**, **MedSAM2**; every chart filters the others.",
        "Median tumour 59 mL; **85%** of scans are after surgery; **73%** have several separate pieces.",
        "Tumours split evenly between the left (48%) and right (47%) brain.",
        "Used to check the split, find unusual patients and see where each model fails.",
    ], size=16, bullet=True, space=9)
    takeaway(s, "Understanding sizes, sides and pieces explained where the models struggle.")
    notes(s, "Presents the web app built to explore the dataset and the results.",
          "Understanding sizes, sides and pieces explained where the models struggle.",
          ["Heat map: where tumour centres are, seen from above (patient's right on the viewer's left)."])

    # 8 ---- pipeline diagram
    s = new_slide(prs, "The pipeline: YOLO finds, MedSAM2 refines", 8, "Method")
    steps = [("MRI slices", "T1C · T2 · FLAIR\nas one colour picture", PANEL, TEXT),
             ("YOLO11-seg", "every slice, shapes\nwith confidence ≥ 0.05", YOLO_C, "FFFFFF"),
             ("Hint map", "probability → logit × 8\nempty / outside head = 'no'", PANEL, TEXT),
             ("MedSAM2", "start slice, then up & down\nmemory: start + last 6", MS_C, "FFFFFF"),
             ("Mirror check", "run on the mirrored patient\nand average", PANEL, TEXT),
             ("Final 3D mask", "> 50% = tumour\ndrop pieces < 100 voxels", PANEL, TEXT)]
    bw, bh, gap, top = Inches(1.78), Inches(1.9), Inches(0.29), Inches(2.0)
    for i, (head, sub, fill, ink) in enumerate(steps):
        l = Inches(0.6) + i * (bw + gap)
        b = box(s, l, top, bw, bh, fill=fill, line=LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE)
        b.adjustments[0] = 0.08
        text(s, l + Inches(0.08), top + Inches(0.15), bw - Inches(0.16), Inches(0.45), f"**{head}**", size=16, color=ink, align=PP_ALIGN.CENTER)
        text(s, l + Inches(0.08), top + Inches(0.7), bw - Inches(0.16), Inches(1.1), sub, size=12, color=ink, align=PP_ALIGN.CENTER)
        if i < len(steps) - 1:
            a = box(s, l + bw + Inches(0.03), top + bh / 2 - Inches(0.14), gap - Inches(0.06), Inches(0.28), fill=MUTED, shape=MSO_SHAPE.RIGHT_ARROW)
    text(s, Inches(0.6), Inches(4.25), Inches(12.1), Inches(2.0), [
        "**Stage 1 (YOLO)** runs once per patient; its hint maps are saved.",
        "**Stage 2 (MedSAM2)** gets the picture **and** YOLO's hint on every slice, plus its memory of the slices it just did.",
        "Only at the very end is the 3D mask compared with the expert's (3D Dice).",
    ], size=17, bullet=True, space=10)
    takeaway(s, "MedSAM2 doesn't search from scratch; it corrects YOLO's guess using the neighbouring slices.")
    notes(s, "Walks through the full inference pipeline from MRI slices to the final 3D mask.",
          "MedSAM2 doesn't search from scratch; it corrects YOLO's guess using the neighbouring slices.",
          ["Start slice = the slice where YOLO is most confident (area × score).",
           "Mirror check: the mirrored copy runs in the same batch, so it costs almost nothing."])

    # 9 ---- YOLO
    s = new_slide(prs, "Stage 1: fine-tuning YOLO", 9, "Method · Results")
    picture(s, "yo_eval_top.png", Inches(0.6), Inches(1.5), Inches(7.6), Inches(4.7))
    text(s, Inches(8.5), Inches(1.6), Inches(4.2), Inches(4.6), [
        "yolo11m-seg at 512 px, trained on the slices of 471 scans, **including empty ones** (teaches when to draw nothing).",
        "Best round picked on YOLO val; stopped early at round 32 of 60.",
        "Extras when predicting: confidence 0.05, mirror averaging, remove pieces < 200 voxels (+0.006 on val).",
        "**Test 3D Dice 0.8834** (median 0.909); found 93.8% of tumour slices.",
    ], size=16, bullet=True, space=10)
    takeaway(s, "A strong 2D baseline: 0.883 3D Dice on 324 unseen test patients.")
    notes(s, "Describes how YOLO was fine-tuned and how well it does on its own.",
          "A strong 2D baseline: 0.883 3D Dice on 324 unseen test patients.",
          ["The confidence chart shows lower confidence = more drawn = better 3D Dice, up to 0.05–0.1."])

    # 10 ---- hint
    s = new_slide(prs, "Stage 2: turning YOLO's output into a MedSAM2 hint", 10, "Method")
    picture(s, "ms_viewer.png", Inches(0.6), Inches(1.5), Inches(12.1), Inches(3.0))
    text(s, Inches(0.6), Inches(4.6), Inches(6.0), Inches(1.7), [
        "Hint = YOLO's **probability per pixel** from every shape at confidence ≥ 0.05.",
        "Turned into a **logit × 8** (−44 … +44): MedSAM2's own scale, where 0 = 50%.",
    ], size=16, bullet=True, space=8)
    text(s, Inches(6.8), Inches(4.6), Inches(5.9), Inches(1.7), [
        "Empty slices and outside the head get a firm **'nothing here'** (−20).",
        "Every slice is hinted; memory carries the start slice + the last 6 slices.",
    ], size=16, bullet=True, space=8)
    takeaway(s, "YOLO's soft guess plus 3D memory lets MedSAM2 fix single-slice mistakes.")
    notes(s, "Explains how YOLO's output becomes the prompt MedSAM2 reads on every slice.",
          "YOLO's soft guess plus 3D memory lets MedSAM2 fix single-slice mistakes.",
          ["Picture (left to right): input picture, YOLO's hint, YOLO's own mask, MedSAM2's answer.",
           "YOLO's own mask uses only shapes ≥ 0.25 and pixels > 50%; it is only for comparison."])

    # 11 ---- training
    s = new_slide(prs, "Training MedSAM2 to fix, not copy", 11, "Method")
    picture(s, "ms_training_curves.png", Inches(0.6), Inches(1.5), Inches(7.0), Inches(4.7))
    text(s, Inches(7.9), Inches(1.6), Inches(4.8), Inches(4.7), [
        "500 training scans, clips of 8 slices, 15 rounds; best round checked on 80 val scans.",
        "**Trained:** mask decoder + prompt encoder (lr 1e-4) and memory (1e-5) = 11.7 M weights (30%).",
        "**Frozen:** the image encoder (70% of weights); training it too gave no gain.",
        "**30% of training hints damaged on purpose** (dropped, shifted, grown, shrunk) while the answer stays the expert's.",
        "Loss: Dice + cross-entropy (+ small IoU and object terms).",
    ], size=15, bullet=True, space=9)
    takeaway(s, "Train only the small parts that read the hint and carry memory; keep the big image encoder frozen.")
    notes(s, "Explains what parts of MedSAM2 are trained and how training teaches correction.",
          "Train only the small parts that read the hint and carry memory; keep the big image encoder frozen.",
          ["Check score levels off around round 10 at about 0.90."])

    # 12 ---- results
    s = new_slide(prs, "Results: MedSAM2 beats YOLO's best on test", 12, "Results")
    picture(s, "chart_final_comparison.png", Inches(0.6), Inches(1.5), Inches(6.6), Inches(4.7))
    text(s, Inches(7.5), Inches(1.6), Inches(5.2), Inches(4.7), [
        "**0.8988** vs **0.8834** mean 3D Dice on the same 324 test patients.",
        "Gain **+0.0154** (95% CI +0.012 … +0.019); Wilcoxon p < 10⁻³³.",
        "**185 patients better, 18 worse** (by more than 0.01); the rest about the same.",
        "Found 96.9% of tumour slices and kept 96.4% of empty slices empty.",
        "Settings were picked on val; test was scored once.",
    ], size=16, bullet=True, space=10)
    takeaway(s, "Adding MedSAM2 gives a consistent, statistically clear gain over the best YOLO.")
    notes(s, "Presents the final test results against YOLO's best setup.",
          "Adding MedSAM2 gives a consistent, statistically clear gain over the best YOLO.",
          ["The fair comparison is against 'YOLO best' (0.8834), not 'YOLO alone' (0.8828)."])

    # 13 ---- where
    s = new_slide(prs, "Where it works, and where it struggles", 13, "Results")
    picture(s, "ms_side_size.png", Inches(0.6), Inches(1.45), Inches(6.9), Inches(2.6),
            caption="Mean MedSAM2 3D Dice by side (rows) and tumour size (columns), test")
    picture(s, "ms_patient.png", Inches(0.6), Inches(4.25), Inches(6.9), Inches(2.0))
    text(s, Inches(7.8), Inches(1.6), Inches(4.9), Inches(4.6), [
        "**Size matters, side doesn't:** about 0.67–0.73 for tumours under 10 mL vs about 0.93 for 100 mL and up; left ≈ right.",
        "The **biggest gains** over YOLO are on **small tumours**.",
        "Slice by slice (bottom): MedSAM2 follows the tumour through memory and **rejects YOLO's isolated false alarms**.",
        "Few patients in some groups (e.g. 'both sides': 17), so those squares are noisy.",
    ], size=15, bullet=True, space=10)
    takeaway(s, "Tiny tumours remain the weak spot, and they are where MedSAM2 helps most.")
    notes(s, "Breaks results down by tumour size and side and shows one patient slice by slice.",
          "Tiny tumours remain the weak spot, and they are where MedSAM2 helps most.")

    # 14 ---- lessons
    s = new_slide(prs, "What we tried and what we learned", 14, "Experiments")
    picture(s, "chart_experiments.png", Inches(0.6), Inches(1.5), Inches(6.2), Inches(3.3),
            caption="Settings experiments (val): only the green change was statistically real")
    picture(s, "chart_errors.png", Inches(7.0), Inches(1.5), Inches(5.7), Inches(3.3),
            caption="Where the remaining wrong voxels are (val)")
    text(s, Inches(0.6), Inches(5.15), Inches(12.1), Inches(1.15), [
        "First design (only the start slice prompted, plus extra correction rounds) made things **worse** than YOLO (0.856 vs 0.864, val); hinting **every** slice fixed it.",
        "Most settings changes moved Dice by ≤ 0.001: **96% of the remaining error is at the tumour edge**, not in finding the tumour.",
    ], size=15, bullet=True, space=6)
    takeaway(s, "Design mattered more than tuning; what's left to win is edge precision.")
    notes(s, "Summarises the experiments: the redesign that worked, tuning results, and where errors remain.",
          "Design mattered more than tuning; what's left to win is edge precision.",
          ["The old-design numbers were on the earlier pools, so they are compared within that setup only."])

    # 15 ---- future work
    s = new_slide(prs, "Future work", 15, "Next steps")
    cols = [("Sharper edges", MS_C, ["Average several trained MedSAM2 runs (an ensemble), with no new training.",
                                     "Edge-focused training loss; finer output resolution."]),
            ("Better hints & small tumours", YOLO_C, ["Weigh hint pixels by shape confidence ('max_weighted').",
                                                      "Give YOLO 3D context; focus on tumours under 10 mL."]),
            ("Beyond this project", GOOD, ["Predict the tumour parts (ET, NETC, SNFH), not only the whole tumour.",
                                           "Test on other hospitals' scans; remove the test/train people overlap; measure speed."])]
    for i, (head, col, items) in enumerate(cols):
        l = Inches(0.6 + i * 4.1)
        box(s, l, Inches(1.6), Inches(3.9), Inches(4.4), fill=PANEL)
        box(s, l, Inches(1.6), Inches(3.9), Inches(0.08), fill=col)
        text(s, l + Inches(0.2), Inches(1.8), Inches(3.5), Inches(0.5), f"**{head}**", size=19, color=col)
        text(s, l + Inches(0.2), Inches(2.45), Inches(3.5), Inches(3.5), items, size=16, bullet=True, space=12)
    takeaway(s, "Next gains come from sharper edges and small tumours, then proving it on outside data.")
    notes(s, "Lists the most promising next steps and the wider follow-ups.",
          "Next gains come from sharper edges and small tumours, then proving it on outside data.",
          ["Experts themselves don't agree perfectly, so expect small gains from here."])

    prs.save(OUT)
    print("saved", OUT, len(prs.slides._sldIdLst), "slides")


if __name__ == "__main__":
    build()
