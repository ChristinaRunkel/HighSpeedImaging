import torch

# life cycle:
# input: n-dimensional data, crop size 
# pad the input data for later
    # params: value for constant padding
    # this = ChunkedProcesser(data, chunk_size, padding_value=0)
    # make sure chunk size is multiple of 2 
# create a flat list of overlapping and cropped data 
    # this.crop_data()
# process each chunk with the network
    # probably have an iterable class for this
    # on user end: store the result back
    # for chunk in this:
        # out = net(chunk)
        # this.store_back(out)
# trim off the outside of each chunk
    # could modify chunks right away, but probably do this in one batch to allow user to access full sized outputs
    # can combine this with the next two steps
# combine the flat list back into the full sized volume
# trim off the padding
    # out_data = this.recombine()
class ChunkedProcessor(object):
    def __init__(self, data, chunk_size, mode='constant', padding_value=0, debug=False):
        '''
        Args:
           data: torch.Tensor 
           chunk_size: tuple of ints with same length as dimensions as data
           padding_value: constant value used for padding
           mode: 'constant', 'reflect', 'replicate' or 'circular'; only constant works arbitrary dimensions, see torch.nn.functional.pad
        '''
        if (torch.IntTensor(chunk_size)%2).sum() > 0:
            raise ValueError("all of chunk_size must be divisible by 2")
        self.crop_size = chunk_size
        self.unpadded_shape = squeeze_shape(data)
        self.data = torch.nn.functional.pad(data, calc_padding(squeeze_shape(data), crop_size=chunk_size), mode=mode, value=padding_value)
        self.data = self.data.squeeze()  # data may be higher dimensional to make padding work
        self.crop_list = []
        self.output_list = []
        self.debug = debug
        self.__crop_data__()
        
    def __crop_data__(self):
        starts = create_crop_starts(self.data.shape, crop_size=self.crop_size, overlap=True, padding=False, flip=True, debug=self.debug)
        self.crop_list = [targeted_crop(self.data, size=self.crop_size, start_index=s) for s in starts]
    
    def __len__(self):
        l = len(self.crop_list)
        return l
    
    def __getitem__(self, index):
        return self.crop_list[index]
    
    def store_back(self, chunk):
        self.output_list += [chunk]
        
    def recombine(self):
        if len(self.output_list) != len(self.crop_list):
            raise RuntimeError("expected {} chunks but {} were received with store_back()".format(
                len(self.output_list), 
                len(self.crop_list)))
            
        crop_width = torch.IntTensor(self.crop_size)//2
        crop_start = crop_width//2
        cropped_outputs = [targeted_crop(o, crop_width, crop_start) for o in self.output_list]
        stitched = combine_cropped(cropped_outputs, self.data.shape, flip=True)
        stitched = targeted_crop(stitched, self.unpadded_shape)
        return stitched


size_tensor = lambda x: torch.Tensor(list(x.shape)) if isinstance(x, torch.Tensor) else torch.Tensor(list(x))
size_tuple = lambda x: tuple(x.tolist())
is_border = lambda indices, full_crops: 0 in indices or True in [True for i,n in zip(indices, lengths) if n-i == 1]
squeeze_shape = lambda x: tuple([y for y in list(x.shape) if y > 1])

# calculate 6d padding sizes and shape into tuple with appropriate ordering (l r t b f ba)
def calc_padding(video, crop_size=(2,100,100)):
    cs = torch.Tensor(crop_size).int()
    t2 = size_tensor(video).int()
    tl = cs//4
    top_left_pad = t2+tl
    unused = top_left_pad%cs
    # desired extension on right side is 1/4 of crop size (just like on the left)
    # `cs-unused` gives perfect crop patches, but does not guarantee this extension
    # most of the time less is needed, but in the worst case we need to be out by 1/2 of crop size to ensure one more patch
    # i.e. we had to add one full crop patch to the left and right
    br = cs-unused+cs//2
    br[unused == 0] = 0
    padding = [[x,y] for x,y in zip(tl.flip(-1).tolist(), br.flip(-1).tolist())]
    padding = tuple([item for sublist in padding for item in sublist])
    return padding

def create_crop_starts(video, crop_size=(2,100,100), overlap=False, padding=False, flip=True, debug=False):
    crop_size = torch.Tensor(crop_size)
    if overlap:
        crop_size /= 2
    crops = size_tensor(video)/crop_size
    if flip:
        crops, crop_size = crops.flip(-1), crop_size.flip(-1)
    #extra_crop = (crops % 1).ceil().type(torch.bool)
    full_crops = crops.int()
    if overlap:
        if padding:
            full_crops += 1
        else:
            full_crops -= 1
    crop_size = crop_size.int()
    offset = crop_size if overlap and padding else torch.zeros(1, dtype=torch.int32)
    
    result = [0] * full_crops.prod().item()
    indices = [0] * full_crops.numel()
    for i in range(len(result)):
        for n in range(len(indices)):
            lastindex = indices[n-1] if n > 0 else 0
            dims_exclusive = full_crops[:n].prod().item()
            dim_now = full_crops[n].item()
            indices[n] = ((i-lastindex) // dims_exclusive) % dim_now
        #slices = tuple([slice(x,x+1) for x in indices])
        x = (torch.IntTensor(indices) * crop_size - offset)
        if flip:
            x = x.flip(-1)
        crop_start = tuple(x.tolist())
        if debug:
            print(crop_start)
        
        result[i] = crop_start
    return result

def targeted_crop(video, size=(2, 100, 100), start_index=(0, 0, 0)):
    if isinstance(size, torch.Tensor):
        size = size_tuple(size)
    if isinstance(start_index, torch.Tensor):
        start_index = size_tuple(start_index)
    #random_start_index = [torch.randint(0, i-o+1, (1,)).item() for i,o in zip(video.shape, size)]
    slices = tuple([slice(x,x+y) for x,y in zip(start_index, size)])
    return video[slices]

def simulate_net_output(video, border_fraction=0.33, mode='m', keep_weight=0.4):
    '''
    Args:
        border_fraction: size of the border as a fraction of the image
        mode: border style, m = multiplicative decrease, a = additive decrease, b = black
        keep_weight: applies to m and a, 0 means border is influenced a lot by randomness, 1 means not at all
    '''
    size_tensor = torch.Tensor(list(video.shape))
    ends_left = size_tensor*border_fraction/2
    ends_right = size_tensor-ends_left
    ends_left, ends_right = ends_left.int(), ends_right.int()
    video = video.clone()
    #left_slice = tuple([slice(x) for x in ends_left.tolist()])
    #right_slice = tuple([slice(x,None) for x in ends_right.tolist()])
    center_slice = tuple([slice(x,y) for x,y in zip(ends_left,ends_right)])
    #video[left_slice] = 0
    #video[right_slice] = 0
    #border_slice = torch.ones_like(video, dtype=torch.bool)[center_slice] = False
    border_slice = torch.ones_like(video, dtype=torch.uint8)[center_slice] = 0
    
    if mode == 'b':
        video[border_slice] = 0
    elif mode == 'm':
        rand = torch.rand_like(video)*(1-keep_weight) + keep_weight
        rand[center_slice] = 1
        video *= rand
        video.clamp_max_(1)
    elif mode == 'a':
        rand = torch.rand_like(video)*(1-keep_weight)
        rand[center_slice] = 0
        video -= rand
        video.clamp_min_(0)
    else:
        raise ValueError(mode)
        
    return video

def combine_cropped(crop_list, full_size, flip=True):
    result = torch.Tensor(full_size)
    crop_size = size_tensor(crop_list[0])
    crops = size_tensor(full_size)/crop_size
    #extra_crop = (crops % 1).ceil().type(torch.bool)
    full_crops = crops.int() -1  # without padding we subtracted 1 earlier too
    crop_size = crop_size.int()
    if flip:
        full_crops = full_crops.flip(-1)
        #crop_size = crop_size.flip(-1)
    
    indices = [-1] * full_crops.numel()
    
    for i in range(len(crop_list)):
        for n in range(len(indices)):
            lastindex = indices[n-1] if n > 0 else 0
            dims_exclusive = full_crops[:n].prod().item()
            dim_now = full_crops[n].item()
            indices[n] = ((i-lastindex) // dims_exclusive) % dim_now
        x = torch.IntTensor(indices)
        if flip:
            x = x.flip(-1)
        y = x+1
        x *= crop_size
        y *= crop_size
        
        slices = tuple([slice(x,y) for x,y in zip(x.tolist(),y.tolist())])
        result[slices] = crop_list[i]
    return result
