from skimage import data
from skimage.transform import swirl

from scipy import misc

import matplotlib.pyplot as plt
import numpy

image = numpy.cast["float64"](misc.lena())
swirled = swirl(image, rotation=0, strength=10, radius=120, order=2)

f, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 3))

ax0.imshow(image, cmap=plt.cm.gray, interpolation='none')
ax0.axis('off')
ax1.imshow(swirled, cmap=plt.cm.gray, interpolation='none')
ax1.axis('off')

plt.show()
