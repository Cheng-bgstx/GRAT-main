import os
import numpy as np
import h5py

folder = "/root/data1/fyc/GRAT-main/data/Synapse_preprocess/train_npz/"   

def inspect_array(name, arr):
    print(f"  - {name}: shape={arr.shape}, dtype={arr.dtype}, "
          f"min={arr.min()}, max={arr.max()}")

for fname in os.listdir(folder):
    path = os.path.join(folder, fname)
    if fname.endswith(".npz"):
        print(f"\n=== NPZ file: {fname} ===")
        data = np.load(path)
        for key in data.keys():
            inspect_array(key, data[key])

    elif fname.endswith(".h5") or fname.endswith(".hdf5"):
        print(f"\n=== H5 file: {fname} ===")
        with h5py.File(path, "r") as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    arr = obj[()]
                    inspect_array(name, arr)
            f.visititems(visit)
