# Class presentation — slide outline (15 slides)

Deck: `BrainTumour_YOLO_MedSAM2.pptx` (speaker notes in Hebrew on every slide).
Rebuild: `make_charts.py` then `build_presentation.py` (needs `python-pptx`).
Numbers come from MedSAM2 run `20260930-234255` and YOLO run `20260929-074550`.

| # | Title | Summary (what it communicates) | Key takeaway | Visual |
|---|---|---|---|---|
| 1 | Brain Tumour Segmentation on MRI with YOLO + MedSAM2 | Introduces the project: outlining brain tumours by combining YOLO and MedSAM2. | A fast 2D detector plus a 3D-aware promptable model outline brain tumours. | 4-panel slice (picture, hint, YOLO, MedSAM2) |
| 2 | Why segment brain tumours automatically? | Explains the clinical need: manual outlining is slow and varies between experts. | Automatic 3D segmentation saves expert time and makes measurements consistent. | Expert outline on FLAIR |
| 3 | Each MRI sequence shows a different part of the tumour | One patient (BraTS-GLI-00046-101, slice 115) in T1, T1C, T2 and FLAIR, each outlining the part it shows best, plus the expert labels and how 3D Dice scores a result. | No single sequence shows the whole tumour, so the models look at several sequences together. | 5 zoomed panels (`make_modalities.py`) |
| 4 | Two models with different strengths | Introduces YOLO11-seg (2D, fast) and SAM2/MedSAM2 (promptable, video memory). | YOLO finds the tumour on a slice; MedSAM2 refines it using the slices around it. | Two comparison cards |
| 5 | Research question and ground rules | States the question and the rules that keep the comparison fair. | A fair two-stage test: does adding MedSAM2 beat the best YOLO we could build? | Question banner + two lists |
| 6 | Data: 1,621 scans from 731 people, split by person | Shows the five pools and why the two models never share people. | Keeping the two models' people apart makes MedSAM2's training hints realistic. | Split table (UI) |
| 7 | Knowing the data: an interactive analytics app | Presents the 4-page web app and key dataset facts. | Understanding sizes, sides and pieces explained where the models struggle. | Analytics KPIs + location heat map (UI) |
| 8 | The pipeline: YOLO finds, MedSAM2 refines | Walks through the inference pipeline from MRI slices to the final 3D mask. | MedSAM2 corrects YOLO's guess using neighbouring slices instead of searching from scratch. | 6-step flow diagram |
| 9 | Stage 1: fine-tuning YOLO | How YOLO was trained and its stand-alone result (test 0.8834). | A strong 2D baseline: 0.883 3D Dice on 324 unseen test patients. | YOLO results (UI) |
| 10 | Stage 2: turning YOLO's output into a MedSAM2 hint | How the probability map becomes a logit hint on every slice, plus memory. | YOLO's soft guess plus 3D memory lets MedSAM2 fix single-slice mistakes. | "See a slice" panel (UI) |
| 11 | Training MedSAM2 to fix, not copy | Which parts are trained, learning rates, and deliberate hint damage. | Train the small parts that read the hint and carry memory; keep the image encoder frozen. | Training curves (UI) |
| 12 | Results: MedSAM2 beats YOLO's best on test | 0.8988 vs 0.8834, +0.0154 (95% CI +0.012…+0.019), 185 better / 18 worse. | Adding MedSAM2 gives a consistent, statistically clear gain over the best YOLO. | Bar chart |
| 13 | Where it works, and where it struggles | Results by tumour size and side, plus one patient slice by slice. | Tiny tumours remain the weak spot, and they are where MedSAM2 helps most. | Side × size grid + patient profile |
| 14 | What we tried and what we learned | The redesign that worked, settings tests, and where errors remain (96% at edges). | Design mattered more than tuning; what's left to win is edge precision. | Experiments chart + error breakdown |
| 15 | Future work | Sharper edges, better hints and small tumours, parts and outside validation. | Next gains come from sharper edges and small tumours, then proving it on outside data. | Three cards |

## Files

- `images/` — UI screenshots (cropped from `images/raw/`), MRI slices from the viewer, and charts from `make_charts.py`.
- `make_charts.py` — draws `chart_*.png` from the saved run results.
- `build_presentation.py` — builds the deck; Hebrew notes are in `HE_NOTES`.
