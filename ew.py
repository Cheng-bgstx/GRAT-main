# debug_count_labels.py
import os
import h5py
import numpy as np
from glob import glob


volume_path = '/root/data1/fyc/GRAT-main/data/Synapse_preprocess/test_vol_h5/'  
list_dir = '/root/data1/fyc/GRAT-main/data/Synapse_preprocess/lists/' 


print("Searching h5/npz files under:", volume_path)
files = glob(os.path.join(volume_path, '**', '*.*'), recursive=True)
files = [f for f in files if f.endswith(('.h5', '.hdf5', '.npz', '.npy'))]
print("Found", len(files), "files (h5/npz/npy)")


class_counts = {i:0 for i in range(1,9)}  
file_missing_label = []

for f in files:
    try:
        if f.endswith('.h5') or f.endswith('.hdf5'):
            with h5py.File(f, 'r') as ff:
                if 'label' in ff:
                    lab = ff['label'][()]
                elif 'label_data' in ff:
                    lab = ff['label_data'][()]
                else:
                    print("No label dataset in", f)
                    file_missing_label.append(f)
                    continue
        elif f.endswith('.npz'):
            d = np.load(f)
            if 'label' in d:
                lab = d['label']
            else:
                print("No label in npz:", f)
                file_missing_label.append(f)
                continue
        elif f.endswith('.npy'):
            # If paired image/label naming unknown, skip
            continue
        else:
            continue

        unique = np.unique(lab)
        for c in range(1,9):
            if c in unique:
                class_counts[c] += 1

    except Exception as e:
        print("Err reading", f, e)

print("Per-class presence in files (number of files containing class):")
for c in range(1,9):
    print(f" Class {c}: {class_counts[c]} files")

if file_missing_label:
    print("\nFiles missing label arrays:", len(file_missing_label))
    for i, ff in enumerate(file_missing_label[:10],1):
        print(" ", i, ff)
