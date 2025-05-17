import numpy as np
import nibabel as nib   
import matplotlib.pyplot as plt

path = "./BraTS20_Training_001/BraTS20_Training_001_seg.nii"

# return shape (240, 240, 155)
def load_image(path):
    img = nib.load(path)
    return img.get_fdata()

load = load_image(path)

print(load)

'''
vmin = np.min(image_data)
vmax = np.max(image_data)
init_slice = image_data.shape[2] // 2
plt.imshow(image_data[:, :, init_slice], vmin=vmin, vmax=vmax)  # can add cmap='gray' 
plt.show()
'''



