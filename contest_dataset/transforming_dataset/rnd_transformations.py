import numpy
import skimage.transform as st

from scipy import ndimage

def translate(xy, xtranslate, ytranslate):
    xy[:, 0] += xtranslate
    xy[:, 1] += ytranslate
    return xy

class TransformationType:
    AFFINE = "Affine"
    EFFECTS = "Effects"

class Transformations(object):
    def __init__(self, transformations=None, n_transformations=3, dtype="float32", rng=None):
        if rng is None:
            self.rng = numpy.random.RandomState(1234)
        else:
            self.rng = rng

        self.transformations = transformations
        self.n_transformations = n_transformations
        self.dtype=dtype

    def get_random_transformations(self):
        pass

    def perform_random_transformations(self):
        pass

    def perform_all_transformations(self):
        pass

class AffineTransformations(Transformations):

    def __init__(self, scaleRatio=0.95, xtrans_bound=None, ytrans_bound=None, rng=None):
        self.scaleRatio = scaleRatio
        if xtrans_bound is None:
            self.xtrans_bound = [-5, 5]
        else:
            self.xtrans_bound = xtrans_bound

        if ytrans_bound is None:
            self.ytrans_bound = [-5, 5]
        else:
            self.ytrans_bound = ytrans_bound
        self.transformations = ["scale", "rotate", "mirror", "translate"]
        super(AffineTransformations, self).__init__()

    def translate_images(self, data, xtrans=0, ytrans=0):
        map_args = {
                "xtrans": xtrans,
                "ytrans": ytrans,
                }
        if data.ndim == 3:
            for i in xrange(data[0].shape[0]):
                data[i] = st.warp(data, translate, map_args=map_args)
        else:
            data = st.warp(data, translate, map_args=map_args)
        return data

    def random_translate_images(self, data, xtrans_r=None, ytrans_r=None):
        if xtrans_r is None:
            if self.xtrans_bound is None:
                xtrans_r = [-5, 5]
            else:
                xtrans_r = self.xtrans_bound
        if ytrans_r is None:
            if self.ytrans_bound is None:
                ytrans_r = [-5, 5]
            else:
                ytrans_r = self.ytrans_bound
        if data.ndim == 3:
            for i in xrange(data[0].shape[0]):
                xtrans = self.rng.random_integers(xtrans_r[0], xtrans_r[1])
                ytrans = self.rng.random_integers(ytrans_r[0], ytrans_r[1])

                map_args = {
                    "xtranslate": xtrans,
                    "ytranslate": ytrans,
                }
                data[i] = st.warp(data, translate, map_args=map_args)
        else:
            xtrans = self.rng.random_integers(xtrans_r[0], xtrans_r[1])
            ytrans = self.rng.random_integers(ytrans_r[0], ytrans_r[1])

            map_args = {
                "xtranslate": xtrans,
                "ytranslate": ytrans,
            }
            data = st.warp(data, translate, map_args=map_args)
            return data

    def rotate_images_90(self, data, rotate_times=-1):
        """Rotate image 90 degrees in the counter clockwise direction, in a randomly determined
        number of times."""
        if rotate_times==-1:
            return data
        rotated_imgs = []
        if data.ndim == 3:
            for img in data:
                img = numpy.rot90(img, rotate_times)
                rotated_imgs.append(img)
            rotated_imgs = numpy.asarray(rotated_imgs)

        else:
            img = numpy.rot90(img, rotate_times)
            rotated_imgs = img
        return rotated_imgs

    def rotate_images(self, data, angle=-1):
        if angle == -1:
            angle = self.rng.random_integers(360)
        if data.ndim <= 2:
            data = ndimage.rotate(data, angle=angle)
        else:
            for i in xrange(data.shape[0]):
                data[i] = ndimage.rotate(data[i], angle=angle)
        return data

    def random_scale_images(self, data, scaleRatio=-1):
        if data.ndim == 3:
            for i in xrange(data.shape[0]):
                if scaleRatio <= 0:
                    scaleRatio = self.rng.uniform(0.85, 1.15)
                scale_dim = self.rng.random_integers(1)
                if scale_dim:
                    #Scale in first dimension
                    img_scaled = st.rescale(data[i], scale=(scaleRatio, 1))
                    scaled_shape = img_scaled.T.shape
                    I = numpy.eye(scaled_shape[0], scaled_shape[1])
                    data[i] = numpy.dot(I, img_scaled)
                else:
                    #Scale in the second dimension
                    img_scaled = st.rescale(data[i], scale=(1, scaleRatio))
                    scaled_shape = img_scaled.T.shape
                    I = numpy.eye(scaled_shape[0], scaled_shape[1])
                    data[i] = numpy.dot(img_scaled, I)
        else:
            if scaleRatio <= 0:
                 scaleRatio = self.rng.uniform(0.85, 1.15)
            scale_dim = self.rng.random_integers(1)
            if scale_dim:
                #Scale in first dimension
                img_scaled = st.rescale(data, scale=(scaleRatio, 1))
                scaled_shape = img_scaled.T.shape
                I = numpy.eye(scaled_shape[0], scaled_shape[1])
                data = numpy.dot(I, img_scaled)
            else:
                #Scale in the second dimension
                img_scaled = st.rescale(data, scale=(1, scaleRatio))
                scaled_shape = img_scaled.T.shape
                I = numpy.eye(scaled_shape[0], scaled_shape[1])
                data = numpy.dot(img_scaled, I)
        return data

    def scale_images(self, data, scaleRatio=-1):
        if scaleRatio <= 0:
            scaleRatio = self.scaleRatio
        if data.ndim == 3:
            for i in xrange(data.shape[0]):
                scale_dim = self.rng.random_integers(1)
                if scale_dim:
                    #Scale in first dimension
                    img_scaled = st.rescale(data[i], scale=(scaleRatio, 1))
                    scaled_shape = img_scaled.T.shape
                    I = numpy.eye(scaled_shape[0], scaled_shape[1])
                    data[i] = numpy.dot(I, img_scaled)
                else:
                    #Scale in the second dimension
                    img_scaled = st.rescale(data[i], scale=(1, scaleRatio))
                    scaled_shape = img_scaled.T.shape
                    I = numpy.eye(scaled_shape[0], scaled_shape[1])
                    data[i] = numpy.dot(img_scaled, I)
        else:
            scale_dim = self.rng.random_integers(1)
            if scale_dim:
                #Scale in first dimension
                img_scaled = st.rescale(data, scale=(scaleRatio, 1))
                scaled_shape = img_scaled.T.shape
                I = numpy.eye(scaled_shape[0], scaled_shape[1])
                data = numpy.dot(I, img_scaled)
            else:
                #Scale in the second dimension
                img_scaled = st.rescale(data, scale=(1, scaleRatio))
                scaled_shape = img_scaled.T.shape
                I = numpy.eye(scaled_shape[0], scaled_shape[1])
                data = numpy.dot(img_scaled, I)
        return data

    def horizontal_flip(self, data):
        flipped_imgs = []
        if data.ndim == 3:
            for img in data:
                img = numpy.flipud(img)
                flipped_imgs.append(img)
            flipped_imgs = numpy.asarray(flipped_imgs, dtype=self.dtype)
        else:
            img = numpy.flipud(img)
            flipped_imgs = img
        return flipped_imgs

    def vertical_flip(self, data):
        flipped_imgs = []
        if data.ndim == 3:
            for img in data:
                img = numpy.fliplr(img)
                flipped_imgs.append(img)
            flipped_imgs = numpy.asarray(flipped_imgs, dtype=self.dtype)
        else:
            img = numpy.fliplr(data)
            flipped_imgs = numpy.asarray(data, dtype=self.dtype)
        return flipped_imgs

class ImageEffects(Transformations):
    def __init__(self):
        self.transoformations = ["swirl", "blur", "sharpen"]
        super(ImageEffects, self).__init__()

    def swirl_images(self, data, strength, radius=100, center=(24, 24)):
        if data.ndim == 3:
            for i in xrange(data.shape[0]):
                img_swirled = st.swirl(data[i], center=center, strength=strength, radius=radius)
                data[i] = img_swirled
        else:
            img_swirled = st.swirl(data, center=center, strength=strength, radius=radius,
                    rotation=0, order=2)
            data = img_swirled
        return data

    def blur_images(self, data, blur_rate=3):
        if data.ndim == 3:
            for i in xrange(data.shape[0]):
                img_blurred = ndimage.gaussian_filter(data[i], sigma=blur_rate)
                data[i] = img_blurred
        else:
            img_blurred = ndimage.gaussian_filter(data, sigma=blur_rate)
            data = img_blurred
        return data

    def enhance_sharpness(self, data):
        if data.ndim == 3:
            for i in xrange(data.shape[0]):
                img_blurred_l = ndimage.gaussian_filter(data[i], sigma=1)
                alpha=30
                data[i] = img_blurred_l + alpha * (img_blurred_l - data[i])
        else:
            img_blurred_l = ndimage.gaussian_filter(data, sigma=1)
            alpha = 30
            data = img_blurred_l + alpha * (img_blurred_l - data)
        return data

class ImageTransformer(AffineTransformations, ImageEffects):

    def __init__(self,
            scaleRatio=0.95,
            xtrans_bound=None,
            ytrans_bound=None,
            transformations=None,
            enable_random_transform=True,
            rng=None):

        super(ImageTransformer, self).__init__(scaleRatio=scaleRatio, xtrans_bound=xtrans_bound,
                ytrans_bound=ytrans_bound, rng=rng)
        if transformations is None:
            self.transformations = [TransformationType.AFFINE, TransformationType.EFFECTS],
        else:
            self.transformations = transformations
        self.enable_random_transform = enable_random_transform
        if rng is None:
            self.rng = numpy.random.RandomState(1234)
        else:
            self.rng = rng

