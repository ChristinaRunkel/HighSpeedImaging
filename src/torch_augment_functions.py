import torch

def to_tensor(uint_array, device=None, normalize=True):
    '''
    input can be anything that torch.as_tensor can handle
    if normalize is true, the result is divided by 255
    '''
    x = torch.as_tensor(uint_array, dtype=torch.float32, device=device)
    return x/255 if normalize else x

def generate_x_bit_sequence(video, x=1, normalized=True):
    '''
    x: the number of bits, not the number of images to average
    normalized: True means the video is in the range [0,1], False for [0,255]
    '''
    assert isinstance(video, torch.Tensor) and video.dtype == torch.float32, "invalid video type"
    
    nvideo = video if normalized else video/255
    num_samples = 2**x-1
    result = torch.zeros_like(video, dtype=torch.float32)
    for _ in range(num_samples):
        result += (nvideo > torch.rand_like(video)).float()
    result /= num_samples
    return result if normalized else result*255
    
def salt_pepper_noise(video, p=0.01, mode='b', normalized=True):
    '''
    p: probability for either salt or pepper to occur, e.g. p/2 for just salt
    mode: b = both, s = salt, p = pepper
    normalized: True means value for salt pixels is 1, otherwise 255
    '''
    max_val = 1. if normalized else 255.
    r = torch.rand(video.shape)
    out = video.clone()
    if mode == 'b' or mode == 'p':
        out[r < p/2] = 0
    if mode == 'b' or mode == 's':
        out[r > 1-p/2] = max_val
    return out

def random_rotate90(video, dims=(1,2), p=1.):
    if torch.rand(1).item() > p:
        return video
    k = torch.randint(4, (1,)).item()
    return torch.rot90(video, k, dims)

def random_hflip(video, dims=(1,), p=0.5):
    if torch.rand(1).item() > p:
        return video
    return torch.flip(video, dims)

def random_crop(video, size=(30, 100, 100)):
    random_start_index = [torch.randint(0, i-o+1, (1,)).item() for i,o in zip(video.shape, size)]
    slices = tuple([slice(x,x+y) for x,y in zip(random_start_index, size)])
    return video[slices]

def targeted_crop(video, size=(30, 100, 100), start_index=(0, 0, 0)):
    slices = tuple([slice(x,x+y) for x,y in zip(start_index, size)])
    return video[slices]
