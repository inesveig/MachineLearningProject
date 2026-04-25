import numpy as np                          #For multi-dim tab
import matplotlib.pyplot as plt
import os                                   #To access files

from scipy.nimage import convolve           #To add the padding of 0
from PIL import Image                       #To manipulate images (open, modify, save)

plt.use('TKAgg')                            #To print back-end graph (bcs here there's no default visualization)



img_path = os.path.join("pt23-cat-picture","cat_pictire.png")
img = Image.open(img_path).convert("L")     #L is the automatique gayscale
img_array = np.array(img)/255


K1 = (1/9) * np.ones((3,3))
K2 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
K3 = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])
K4 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
K5 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
K6 = np.array([[-2, -1, 0], [-1, 1, -1], [0, 1, 2]])














