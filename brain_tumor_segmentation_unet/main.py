import numpy as np
import nibabel as nib   
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

image_path = "./BraTS20_Training_001/BraTS20_Training_001_t1.nii"
image_obj = nib.load(image_path)

image_data = image_obj.get_fdata()

vmin = np.min(image_data)
vmax = np.max(image_data)


def explore_with_slider():
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    init_slice = image_data.shape[2] // 2
    l = plt.imshow(image_data[:, :, init_slice], cmap='gray', vmin=vmin, vmax=vmax)

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Layer', 0, image_data.shape[2]-1, valinit = init_slice, valfmt='%0.0f')

    def update(val):
        layer = int(slider.val)
        l.set_data(image_data[:, :, layer])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

explore_with_slider()