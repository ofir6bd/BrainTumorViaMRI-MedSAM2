# Class presentation — slide outline (21 slides)

Deck: `BrainTumour_YOLO_MedSAM2.pptx` (speaker notes in Hebrew on every slide).
Rebuild: `make_charts.py` then `build_presentation.py` (needs `python-pptx`).
Numbers come from MedSAM2 run `20260930-234255` and YOLO run `20260929-074550`.

| # | Title | Summary (what it communicates) | Key takeaway | Visual |
|---|---|---|---|---|
| 1 | Brain Tumour Segmentation on MRI with YOLO + MedSAM2 | Introduces the project: outlining brain tumours by combining YOLO and MedSAM2. | A fast 2D detector plus a 3D-aware promptable model outline brain tumours. | 4-panel slice (picture, hint, YOLO, MedSAM2) |
| 2 | Why segment brain tumours automatically? | Explains the clinical need: manual outlining is slow and varies between experts. | Automatic 3D segmentation saves expert time and makes measurements consistent. | Expert outline on FLAIR |
| 3 | Each MRI sequence shows a different part of the tumour | One patient (BraTS-GLI-00046-101, slice 115) in T1, T1C, T2 and FLAIR, each outlining the part it shows best, plus the expert labels and how 3D Dice scores a result. | No single sequence shows the whole tumour, so the models look at several sequences together. | 5 zoomed panels (`make_modalities.py`) |
| 4 | Model 1 · YOLO11-seg: one slice in, shapes out | What YOLO receives (one colour slice) and returns (box, confidence, mask per shape) on a real slice. | YOLO turns each slice into shapes with confidences, fast, but blind to the slices around it. | Input → model → output (`make_model_io.py`) |
| 5 | Model 2 · MedSAM2: three inputs, one corrected mask | MedSAM2's inputs (picture, YOLO hint, memory of previous slices) and its output (mask + 'tumour here?' score). | MedSAM2 combines the picture, YOLO's hint and its memory into a corrected mask per slice. | Inputs → model → output + score card |
| 6 | Research question and ground rules | States the question and the rules that keep the comparison fair. | A fair two-stage test: does adding MedSAM2 beat the best YOLO we could build? | Question banner + two lists |
| 7 | Data: 1,621 scans from 731 people, split by person | Shows the five pools and why the two models never share people. | Keeping the two models' people apart makes MedSAM2's training hints realistic. | Split table (UI) |
| 8 | Knowing the data: an interactive analytics app | Presents the 4-page web app and key dataset facts. | Understanding sizes, sides and pieces explained where the models struggle. | Analytics KPIs + location heat map (UI) |
| 9 | The pipeline: YOLO finds, MedSAM2 refines | Walks through the inference pipeline from MRI slices to the final 3D mask. | MedSAM2 corrects YOLO's guess using neighbouring slices instead of searching from scratch. | 6-step flow diagram |
| 10 | Stage 1: fine-tuning YOLO | How YOLO was trained and its stand-alone result (test 0.8834). | A strong 2D baseline: 0.883 3D Dice on 324 unseen test patients. | YOLO results (UI) |
| 11 | Stage 2: turning YOLO's output into a MedSAM2 hint | How the probability map becomes a logit hint on every slice, plus memory. | YOLO's soft guess plus 3D memory lets MedSAM2 fix single-slice mistakes. | "See a slice" panel (UI) |
| 12 | The hint, step by step | On patient BraTS-GLI-00046-101 (slice 115): A store probability 0–255, B logit, C × 8, D 'nothing here' = −20 for empty slices and outside the head; histogram of all the patient's probabilities and the conversion table. | The hint = YOLO's confidence per pixel on MedSAM2's scale, plus a firm 'no' where YOLO saw nothing. | 3 hint panels + histogram (`make_hint_steps.py`, `make_hint_vs_mask.py`, `make_mirror.py`) + table |
| 13 | YOLO's hint vs YOLO's own mask | On slice 120 of the same patient: the soft hint (all shapes ≥ 0.05, probability per pixel) against YOLO's yes/no mask (shapes ≥ 0.25, pixels > 0.5), and what only the hint contains (1,162 px, 99 real tumour). | The hint carries everything YOLO suspects; the own mask is only YOLO's confident final answer. | 3 panels (`make_hint_vs_mask.py`, `make_mirror.py`) + two definition cards |
| 14 | The mirror check | On test patient BraTS-GLI-00008-101 (slice 78): the normal and mirrored patient run in one batch, the mirrored answer is flipped back and averaged per pixel; the two answers differ only at the edges (Dice 0.8457 / 0.8554 / averaged 0.8524). | The mirror check is a cheap second opinion: it steadies the edges, but adds very little Dice (+0.0004 overall). | Inputs → MedSAM2 → 4 answer panels (`make_mirror.py`) + numbers card |
| 15 | Training MedSAM2 to fix, not copy | Which parts are trained, learning rates, and deliberate hint damage. | Train the small parts that read the hint and carry memory; keep the image encoder frozen. | Training curves (UI) |
| 16 | MedSAM2 fine-tuning: which parts we train | The architecture figure, why each part does or doesn't need training, the weights and learning rate per part, and the three training options compared (decoder+prompt / + memory / all). | Train the 30% that reads the hint and carries memory; the big image encoder can stay frozen. | Architecture figure (Medical SAM 2 paper) + two tables |
| 17 | Results: MedSAM2 beats YOLO's best on test | 0.8988 vs 0.8834, +0.0154 (95% CI +0.012…+0.019), 185 better / 18 worse. | Adding MedSAM2 gives a consistent, statistically clear gain over the best YOLO. | Bar chart |
| 18 | Where it works, and where it struggles | Results by tumour size and side, plus one patient slice by slice. | Tiny tumours remain the weak spot, and they are where MedSAM2 helps most. | Side × size grid + patient profile |
| 19 | What we tried and what we learned | The redesign that worked, settings tests, and where errors remain (96% at edges). | Design mattered more than tuning; what's left to win is edge precision. | Experiments chart + error breakdown |
| 20 | Future work | Sharper edges, better hints and small tumours, parts and outside validation. | Next gains come from sharper edges and small tumours, then proving it on outside data. | Three cards |
| 21 | Thank you | Closes the talk with the three headline numbers and invites questions. | MedSAM2 reached 0.8988 on test, +0.0154 over YOLO's best, 185 patients better and 18 worse. | Number cards + labelled tumour image |

## Files

- `images/` — UI screenshots (cropped from `images/raw/`), MRI slices from the viewer, and charts from `make_charts.py`.
- `make_charts.py` — draws `chart_*.png` from the saved run results.
- `make_modalities.py`, `make_model_io.py`, `make_hint_steps.py`, `make_hint_vs_mask.py`, `make_mirror.py` — the slide 3 and slides 4–5 pictures.
- `build_presentation.py` — builds the deck; Hebrew notes are in `HE_NOTES`.
