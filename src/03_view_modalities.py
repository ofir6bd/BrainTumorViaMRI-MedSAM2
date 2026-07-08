import os
import yaml
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.widgets import Button

with open("config.yaml", "r") as _f:
    _CFG = yaml.safe_load(_f)
DATASET_DIR = _CFG["paths"]["dataset"]

LABEL_NAMES  = {1: "NETC", 2: "SNFH", 3: "ET", 4: "RC"}
LABEL_COLORS = {1: 'red', 2: 'green', 3: 'blue', 4: 'yellow'}

_LABEL_OVERLAY = {
    1: (1.00, 0.27, 0.00, 0.50),
    2: (0.00, 1.00, 0.50, 0.50),
    3: (0.00, 0.80, 1.00, 0.50),
    4: (1.00, 0.90, 0.00, 0.50),
}
_LABEL_BBOX = {
    1: '#ff4500',
    2: '#00ff80',
    3: '#00ccff',
    4: '#ffe600',
}

def find_patients_in_training_data():
    dataset_dir = DATASET_DIR
    if not os.path.exists(dataset_dir):
        print(f"Directory not found: {dataset_dir}")
        return []

    patients = []
    for root, _, files in os.walk(dataset_dir):
        nii_files = [f for f in files if f.endswith(('.nii', '.nii.gz'))]
        seg_files = [f for f in nii_files if 'seg' in f.lower()]
        if not nii_files or not seg_files:
            continue

        patient_id = os.path.basename(root)
        if patient_id == "training_data":
            continue

        modalities = {}
        for f in nii_files:
            if 'seg' in f.lower():
                continue
            name = f.lower()
            if '-t1c' in name: modalities['T1C']   = os.path.join(root, f)
            if '-t1n' in name: modalities['T1']    = os.path.join(root, f)
            if '-t2w' in name: modalities['T2']    = os.path.join(root, f)
            if '-t2f' in name: modalities['FLAIR'] = os.path.join(root, f)

        patients.append({
            'patient_id':   patient_id,
            'seg_file':     os.path.join(root, seg_files[0]),
            'modalities':   modalities,
            'display_name': patient_id,
        })

    patients.sort(key=lambda x: x['patient_id'])
    return patients

def _peak_slice_per_label(seg_data):
    result = {}
    for label_id in LABEL_NAMES:
        mask = (seg_data == label_id)
        if not np.any(mask):
            continue
        counts = np.array([mask[:, :, z].sum() for z in range(seg_data.shape[2])])
        peak_z = int(np.argmax(counts))
        result[label_id] = (peak_z, int(counts[peak_z]))
    return result

def _compute_bbox(mask_2d):
    rows = np.any(mask_2d, axis=1)
    cols = np.any(mask_2d, axis=0)
    if not rows.any():
        return None
    r_min = int(np.argmax(rows))
    r_max = int(len(rows) - 1 - np.argmax(rows[::-1]))
    c_min = int(np.argmax(cols))
    c_max = int(len(cols) - 1 - np.argmax(cols[::-1]))
    return r_min, r_max, c_min, c_max

def _pick_background_modality(patient_info):
    for key in ('T1C', 'T1', 'T2', 'FLAIR'):
        if key in patient_info['modalities']:
            return nib.load(patient_info['modalities'][key]).get_fdata(), key
    return None, None

def _norm_bg(bg_vol, z):
    bg_slice = bg_vol[:, :, z].astype(np.float32)
    nz = bg_slice[bg_slice != 0]
    lo = float(np.percentile(nz, 1))  if nz.size else 0
    hi = float(np.percentile(nz, 99)) if nz.size else 1
    return np.clip((bg_slice - lo) / max(hi - lo, 1e-8), 0, 1)

def _make_toggle_button(fig, state_dict, toggleable_artists):
    ax_btn = plt.axes([0.42, 0.02, 0.16, 0.05])
    btn    = Button(ax_btn, 'Hide Labels', color='#3a3a5c', hovercolor='#555580')
    btn.label.set_color('white')
    btn.label.set_fontsize(10)

    def toggle(event):
        state_dict['on'] = not state_dict['on']
        for artist in toggleable_artists:
            if artist is not None:
                artist.set_visible(state_dict['on'])
        btn.label.set_text('Show Labels' if not state_dict['on'] else 'Hide Labels')
        fig.canvas.draw_idle()

    btn.on_clicked(toggle)
    return btn

def plot_modalities_with_seg(patient_info):
    seg_data   = nib.load(patient_info['seg_file']).get_fdata()
    slice_sums = [np.sum(seg_data[:, :, z] > 0) for z in range(seg_data.shape[2])]
    best_z     = int(np.argmax(slice_sums)) if max(slice_sums) > 0 \
                 else seg_data.shape[2] // 2

    colors      = ['black', 'red', 'green', 'blue', 'yellow']
    custom_cmap = ListedColormap(colors)

    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    plt.subplots_adjust(bottom=0.10)
    fig.suptitle(
        f"Patient: {patient_info['patient_id']} | Slice z={best_z}",
        fontsize=16,
    )

    display_list = ['T1', 'T1C', 'T2', 'FLAIR', 'SEG']
    axes_flat    = axes.flatten()
    seg_ax       = None

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   label="1: NETC (Non-Enhancing Core)(Red)",
                   markerfacecolor='red',    markersize=8),
        plt.Line2D([0], [0], marker='o', color='w',
                   label="2: SNFH (FLAIR Hyperintensity)(Green)",
                   markerfacecolor='green',  markersize=8),
        plt.Line2D([0], [0], marker='o', color='w',
                   label="3: ET (Enhancing Tumor)(Blue)",
                   markerfacecolor='blue',   markersize=8),
        plt.Line2D([0], [0], marker='o', color='w',
                   label="4: RC (Resection Cavity)(Yellow)",
                   markerfacecolor='yellow', markersize=8),
    ]

    for i, item in enumerate(display_list):
        ax = axes_flat[i]
        if item == 'SEG':
            seg_ax    = ax
            slice_seg = seg_data[:, :, best_z]
            ax.imshow(slice_seg, cmap=custom_cmap, vmin=0, vmax=4)
            ax.set_title("BraTS 2024 Challenge Colors")
            unique_labels = np.unique(slice_seg)
            active_legend = [le for idx, le in enumerate(legend_elements, 1)
                             if idx in unique_labels]
            if active_legend:
                ax.legend(handles=active_legend, loc='lower left',
                          fontsize='x-small', frameon=True)
        elif item in patient_info['modalities']:
            img_data = nib.load(patient_info['modalities'][item]).get_fdata()
            ax.imshow(img_data[:, :, best_z], cmap='gray')
            ax.set_title(item)
        ax.axis('off')

    axes_flat[-1].axis('off')

    _btn = _make_toggle_button(fig, {'on': True}, [seg_ax])
    plt.show()

def plot_bbox_per_label(patient_info):
    seg_data = nib.load(patient_info['seg_file']).get_fdata()
    peaks    = _peak_slice_per_label(seg_data)

    if not peaks:
        print("  No labels found in segmentation.")
        return

    bg_vol, bg_name = _pick_background_modality(patient_info)
    n_labels        = len(peaks)

    n_cols = int(np.ceil(np.sqrt(n_labels)))
    n_rows = int(np.ceil(n_labels / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(8 * n_cols, 8 * n_rows),
        squeeze=False,
        facecolor='#1a1a2e',
    )
    plt.subplots_adjust(bottom=0.08, hspace=0.12, wspace=0.06)
    fig.suptitle(
        f"BBox per Label  |  Patient: {patient_info['patient_id']}\n"
        f"Background: {bg_name or 'none'}  |  Each panel = peak slice for that label",
        fontsize=14, fontweight='bold', color='white',
    )

    toggleable = []

    for idx, (label_id, (peak_z, px_count)) in enumerate(sorted(peaks.items())):

        row = idx // n_cols
        col = idx %  n_cols

        label_name   = LABEL_NAMES[label_id]
        overlay_rgba = _LABEL_OVERLAY[label_id]
        bbox_color   = _LABEL_BBOX[label_id]
        mask_2d      = (seg_data[:, :, peak_z] == label_id)
        bbox         = _compute_bbox(mask_2d)

        bg_norm = _norm_bg(bg_vol, peak_z) if bg_vol is not None \
                  else np.zeros(seg_data.shape[:2])

        ax = axes[row, col]
        ax.set_facecolor('black')
        ax.imshow(bg_norm, cmap='gray', vmin=0, vmax=1)

        overlay = np.zeros((*mask_2d.shape, 4), dtype=float)
        overlay[mask_2d] = overlay_rgba
        ov_img = ax.imshow(overlay, interpolation='none')
        toggleable.append(ov_img)

        if bbox:
            r_min, r_max, c_min, c_max = bbox
            w = c_max - c_min
            h = r_max - r_min

            rect = mpatches.Rectangle(
                (c_min, r_min), w, h,
                linewidth=2.5, edgecolor=bbox_color,
                facecolor='none', linestyle='--',
            )
            ax.add_patch(rect)
            toggleable.append(rect)

            txt = ax.text(
                c_min, max(r_min - 6, 0),
                f"z={peak_z}  px={px_count:,}  W={w}  H={h}",
                color='white', fontsize=8, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.55, pad=2, edgecolor='none'),
            )
            toggleable.append(txt)

        ax.set_title(
            f"Label {label_id}: {label_name}",
            color=bbox_color, fontsize=10, fontweight='bold',
        )
        ax.axis('off')
        for sp in ax.spines.values():
            sp.set_edgecolor(bbox_color)
            sp.set_linewidth(1.5)

    for idx in range(n_labels, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].set_visible(False)

    _btn = _make_toggle_button(fig, {'on': True}, toggleable)
    plt.show()

def select_patient(patients):
    print()
    for i, p in enumerate(patients[:25]):
        print(f"  {i+1:2d}. {p['display_name']}")
    if len(patients) > 25:
        print(f"  ... ({len(patients) - 25} more not shown)")
    try:
        num = int(input(f"\n  Select index (1-{min(len(patients), 25)}): ")) - 1
        if 0 <= num < len(patients):
            return patients[num]
        print("  Index out of range.")
    except ValueError:
        print("  Invalid number.")
    return None

def main():
    patients = find_patients_in_training_data()
    if not patients:
        return

    while True:
        print(f"\n{'='*40}")
        print(f"  BRATS 2024 VIEWER  |  {len(patients)} patients")
        print(f"{'='*40}")
        print("  1. Modalities + Segmentation")
        print("  2. BBox per Label (peak slice)")
        print("  q. Quit")

        choice = input("\n  Choice: ").strip().lower()

        if choice == 'q':
            break
        elif choice == '1':
            p = select_patient(patients)
            if p:
                plot_modalities_with_seg(p)
        elif choice == '2':
            p = select_patient(patients)
            if p:
                plot_bbox_per_label(p)
        else:
            print("  Unknown option.")

if __name__ == "__main__":
    main()
