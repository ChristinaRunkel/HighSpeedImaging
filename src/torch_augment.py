import torch_augment_functions as F


class ToTensor(object):
    '''
    input can be anything that torch.as_tensor can handle
    normalized: True means the input video is in the range [0,255] and will be adjusted to [0,1]
    '''
    def __init__(self, device=None, normalize=True):
        self.device = device
        self.normalize = normalize

    def __call__(self, uint_array):
        return F.to_tensor(uint_array, device=self.device, normalize=self.normalize)

    def __repr__(self):
        return self.__class__.__name__ + '(device='+str(self.device)+', normalize='+str(self.normalize)+')'
    

class GenXbitSequence(object):
    '''
    x: the number of bits, not the number of images to average
    normalized: True means the video is in the range [0,1], False for [0,255]
    '''
    def __init__(self, x=1, normalized=True):
        self.x = x
        self.normalized = normalized

    def __call__(self, video):
        return F.generate_x_bit_sequence(video, self.x, self.normalized)

    def __repr__(self):
        return self.__class__.__name__ + '(x='+str(self.x)+', normalized='+str(self.normalized)+')'


class SaltPepperNoise(object):
    '''
    p: probability for either salt or pepper to occur, e.g. p/2 for just salt
    mode: b = both, s = salt, p = pepper
    '''
    def __init__(self, p=0.01, mode='b', normalized=True):
        self.p = p
        self.mode = mode
        self.normalized = normalized

    def __call__(self, video):
        return F.salt_pepper_noise(video, self.p, self.mode, self.normalized)

    def __repr__(self):
        return self.__class__.__name__ + '(p='+str(self.p)+', mode=\''+self.mode+'\', normalized='+str(self.normalized)+')'


class RandomRotate90(object):
    def __init__(self, dims=(1,2), p=1.):
        self.dims = dims
        self.p = p

    def __call__(self, video):
        return F.random_rotate90(video, self.dims, self.p)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomHflip(object):
    def __init__(self, dims=(1,), p=0.5):
        self.dims = dims
        self.p = p

    def __call__(self, video):
        return F.random_hflip(video, self.dims, self.p)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomCrop(object):
    def __init__(self, size=(30, 100, 100)):
        self.size = size

    def __call__(self, video):
        return F.random_crop(video, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size='+str(self.size)+')'


class TargetedCrop(object):
    def __init__(self, size=(30, 100, 100), start=(0,0,0)):
        self.size = size
        self.start = start

    def __call__(self, video):
        return F.targeted_crop(video, self.size, self.start)

    def __repr__(self):
        return self.__class__.__name__ + '(size='+str(self.size)+', start='+str(self.start)+')'


# Code from torchvision.transforms
class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string