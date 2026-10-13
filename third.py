import glob, nibabel as nib, numpy as np, os
files = sorted(glob.glob('/root/data1/fyc/GRAT-main/data/Synapse_raw/label/*.nii.gz'))
for f in files:
    lab = nib.load(f).get_fdata()
    vals,counts = np.unique(lab, return_counts=True)
    print(os.path.basename(f), dict(zip(vals.astype(int), counts)))
