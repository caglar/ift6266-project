import skimage
import skimage.data as sd
from scipy import ndimage
from scipy import misc
import pylab as pb
import numpy
from skimage import img_as_float

from rnd_transformations import ImageTransformer


transformer = ImageTransformer()
img = numpy.cast["uint8"](misc.lena())

img = transformer.swirl_images(img, strength=4, radius=120, center=(200, 200))
img = transformer.random_translate_images(img, xtrans_r=[-30, 30], ytrans_r=[-30, 30])
img = transformer.rotate_images_90(img)
img = transformer.random_scale_images(img)
img = transformer.blur_images(img, blur_rate=1)
img = transformer.enhance_sharpness(img)

import ipdb; ipdb.set_trace()


pb.gray()
pb.imshow(img)
pb.show()
