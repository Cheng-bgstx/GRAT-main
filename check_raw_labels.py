# check_raw_labels.py
import glob, os
import nibabel as nib
import numpy as np

src_path = '/root/data1/fyc/GRAT-main/data/Synapse_raw/'  
# list of test case numbers in preprocess_data.py
test_data = [1,2,3,4,8,22,25,29,32,35,36,38]

image_files = sorted(glob.glob(os.path.join(src_path, 'img', '*.nii.gz')))
label_files = sorted(glob.glob(os.path.join(src_path, 'label', '*.nii.gz')))

pairs = list(zip(image_files, label_files))
print("Checking raw label unique values for test cases:")
for img_f, lab_f in pairs:
    num = int(os.path.basename(img_f)[3:7])
    if num in test_data:
        lab = nib.load(lab_f).get_fdata()
        print(f"case {num}: unique labels = {np.unique(lab)}")
