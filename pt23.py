import numpy as np                          #For multi-dim tab
import matplotlib.pyplot as plt
import os                                   #To access files

from scipy.ndimage import convolve           #To add the padding of 0
from PIL import Image                       #To manipulate images (open, modify, save)

import matplotlib as mpl
mpl.use('TKAgg')                            #To print back-end graph (bcs here there's no default visualization)



img_path = os.path.join(".\pt23-cat-picture","cat_picture.png")
img = Image.open(img_path).convert("L")     #L is the automatique gayscale
img_array = np.array(img)/255
print("a")

#Filters
K1 = (1/9) * np.ones((3,3))
K2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
K3 = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])
K4 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
K5 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
K6 = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])

print("b")

filters = {"K1" : K1, "K2" : K2, "K3" : K3, "K4" : K4, "K5" : K5, "K6" : K6}


fig, axes = plt.subplots(2, 4, figsize = (18, 9))

axes[0, 0].imshow(img_array, cmap = "gray")
axes[0,0].axis("off")

print("c")

#To apply filters
for ax, (name, K) in zip(axes.flat[1:], filters.items()):
    filtered = convolve(img_array, K)
    ax.imshow(filtered, cmap = "gray")
print("d")


plt.savefig("K-filters_on_cat.png", dpi = 150)
plt.show()

print("Done")








