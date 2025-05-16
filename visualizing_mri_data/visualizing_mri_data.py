import numpy as np
import nibabel as nib   
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

image_path = "./T1.nii.gz"
image_obj = nib.load(image_path)

image_data = image_obj.get_fdata()

print(image_data.shape)

maxval = 290    # including depth
i = np.random.randint(0, maxval)


def explore_with_slider():
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    l = plt.imshow(image_data[:, :, i], cmap='gray')
    ax.set_title('Explore Layers of MRI')

    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Layer', 0, image_data.shape[2]-1, valinit=i, valfmt='%0.0f')

    def update(val):
        layer = int(slider.val)
        l.set_data(image_data[:, :, layer])
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
    

explore_with_slider()

