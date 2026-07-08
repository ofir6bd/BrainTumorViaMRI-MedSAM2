import os
import yaml
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap

with open("config.yaml", "r") as _f:
    _CFG = yaml.safe_load(_f)
DATASET_DIR = _CFG["paths"]["dataset"]

BRATS_COLORS = ['none', 'red', 'green', 'blue', 'yellow']
BRATS_CMAP = ListedColormap(BRATS_COLORS)

def find_patients_in_training_data():
    dataset_dir = DATASET_DIR

    if not os.path.exists(dataset_dir):
        print(f"Training data directory not found: {dataset_dir}")
        return []

    print(f"Searching in: {dataset_dir}")

    patients = []

    for root, dirs, files in os.walk(dataset_dir):
        nii_files = [f for f in files if f.endswith(('.nii', '.nii.gz'))]

        if len(nii_files) >= 2:
            brain_files = []
            seg_files = []

            for file in nii_files:
                if 'seg' in file.lower():
                    seg_files.append(os.path.join(root, file))
                else:
                    brain_files.append(os.path.join(root, file))

            if brain_files and seg_files:
                patient_id = os.path.basename(root)

                if patient_id == "training_data":
                    continue

                for brain_file in brain_files:
                    seg_file = seg_files[0]

                    brain_name = os.path.basename(brain_file).replace('.nii.gz', '').replace('.nii', '')
                    seg_name = os.path.basename(seg_file).replace('.nii.gz', '').replace('.nii', '')

                    patients.append({
                        'patient_id': patient_id,
                        'brain_file': brain_file,
                        'seg_file': seg_file,
                        'brain_name': brain_name,
                        'seg_name': seg_name,
                        'display_name': f"{brain_name}"
                    })

    patients.sort(key=lambda x: (x['patient_id'], x['brain_name']))
    print(f"Found {len(patients)} brain images with segmentation")
    return patients

def plot_3d_scatter(seg_data, title="3D Segmentation Scatter", max_points=3000):
    try:
        print("Creating 3D scatter plot...")

        unique_values = np.unique(seg_data)
        unique_values = unique_values[unique_values > 0]

        if len(unique_values) == 0:
            print("No segmentation data found for 3D plotting")
            return

        print(f"Available segments: {unique_values}")

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        for value in unique_values:
            coords = np.where(seg_data == value)

            if len(coords[0]) > 0:
                if len(coords[0]) > max_points:
                    indices = np.random.choice(len(coords[0]), max_points, replace=False)
                    x, y, z = coords[0][indices], coords[1][indices], coords[2][indices]
                else:
                    x, y, z = coords

                color = BRATS_COLORS[int(value)] if int(value) < len(BRATS_COLORS) else 'purple'

                ax.scatter(x, y, z, c=color, alpha=0.7, s=2,
                          label=f'Label {int(value)} ({len(coords[0])} voxels)')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{title}\nBraTS 2024 Colors: NETC=Red · SNFH=Green · ET=Blue · RC=Yellow')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error creating 3D scatter plot: {e}")

def plot_patient(patient_info):
    try:
        brain_path = patient_info['brain_file']
        seg_path = patient_info['seg_file']
        patient_id = patient_info['patient_id']
        brain_name = patient_info['brain_name']

        print(f"\nLoading: {patient_id} - {brain_name}")

        brain_data = nib.load(brain_path).get_fdata()
        seg_data = nib.load(seg_path).get_fdata()

        print(f"Brain shape: {brain_data.shape}")
        print(f"Seg shape: {seg_data.shape}")
        print(f"Seg values: {np.unique(seg_data)}")

        slice_sums = [np.sum(seg_data[:, :, z] > 0) for z in range(seg_data.shape[2])]
        best_slice = np.argmax(slice_sums) if max(slice_sums) > 0 else seg_data.shape[2] // 2

        print(f"Best slice: {best_slice}")

        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        fig.suptitle(f'{patient_id} - {brain_name} - Slice {best_slice}', fontsize=16, fontweight='bold')

        axes[0].imshow(brain_data[:, :, best_slice], cmap='gray')
        axes[0].set_title('Brain Image')
        axes[0].axis('off')

        axes[1].imshow(seg_data[:, :, best_slice], cmap=BRATS_CMAP, vmin=0, vmax=4)
        axes[1].set_title('BraTS Labels (1:R, 2:G, 3:B, 4:Y)')
        axes[1].axis('off')

        axes[2].imshow(brain_data[:, :, best_slice], cmap='gray')
        axes[2].imshow(seg_data[:, :, best_slice], cmap=BRATS_CMAP, alpha=0.5, vmin=0, vmax=4)
        axes[2].set_title('Tumor Overlay')
        axes[2].axis('off')

        binary_mask = seg_data[:, :, best_slice] > 0
        axes[3].imshow(brain_data[:, :, best_slice], cmap='gray')
        axes[3].imshow(binary_mask, cmap='Reds', alpha=0.4)
        axes[3].set_title('Binary Mask')
        axes[3].axis('off')

        plt.tight_layout()
        plt.show()

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

def main():

    patients = find_patients_in_training_data()

    if not patients:
        print("No patients found!")
        return

    while True:
        print(f"\n=== BRAIN IMAGE VIEWER ===")
        print(f"Found {len(patients)} brain images")
        print("1. View 2D image")
        print("2. View 3D scatter plot (all segments)")
        print("q. Quit")

        choice = input("\nChoice: ").strip().lower()

        if choice == 'q':
            break

        elif choice in ['1', '2']:
            print(f"\nImages (showing first 100):")
            display_patients = patients[:100]

            for i, patient in enumerate(display_patients):
                print(f"{i+1:2d}. {patient['display_name']}")

            if len(patients) > 100:
                print(f"... and {len(patients) - 100} more")

            try:
                num = int(input(f"\nEnter number (1-{min(len(patients), 100)}): ")) - 1
                if 0 <= num < min(len(patients), 100):
                    patient_info = display_patients[num]

                    if choice == '1':
                        plot_patient(patient_info)
                    elif choice == '2':
                        seg_data = nib.load(patient_info['seg_file']).get_fdata()
                        plot_3d_scatter(seg_data, f"{patient_info['patient_id']} - {patient_info['brain_name']}")
                else:
                    print("Invalid number")
            except ValueError:
                print("Invalid input")

        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
