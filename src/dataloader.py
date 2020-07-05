import torch
import torch.utils.data
import torch_augment as ta
import h5py
import numpy as np


'''
Class to load video data 
Original data from 'Deep Video Deblurring for Hand-held Cameras'-Dataset (https://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/#dataset)
Generation of n-bit sequences by Bernoulli sampling, cropping, flipping, rotating + salt/pepper noise
'''
class VideoVolume(torch.utils.data.Dataset):
    def __init__(self, gpu, normalize, size, test=False, path="data/goPro_2p5k_50f_volume.mat", start=0, end=2500, debug=False):
        'Initialization'
        f = h5py.File(path, 'r')
        if debug:
            print(f['#refs#'])
            print(f['images/gnt'])
            print(f['images/set'])
            print(f['selected_Scenes'])
        gnt = list(f['images/gnt'])
        probe_object = f[gnt[0][0]] # role of second index?
        if debug:
            print(probe_object) # shape (64, 160, 160) dtype "|u1"
        # numpy dtype byte order '=': native '<': little-endian '>': big-endian '|': not applicable
        # types: O = python object, f = float, u = unsigned int
        # size: number of bytes (or characters)

        'Train and test transformations'
        both_train_transforms = ta.Compose([
            ta.ToTensor(device=gpu, normalize=normalize),
            ta.RandomCrop(size=size),
            ta.RandomHflip(),
            ta.RandomRotate90(),
        ])
        noise_train_transforms = ta.Compose([
            ta.GenXbitSequence(x=1, normalized=normalize),
            ta.SaltPepperNoise(p=0.02, normalized=normalize),  # additional noise during training might regularize a bit
        ])
        both_test_transforms = ta.Compose([
            ta.ToTensor(device=gpu, normalize=normalize),  # do not augment during testing
            ta.TargetedCrop(size=size), # until stitching works 
        ])
        noise_test_transforms = ta.Compose([
            ta.GenXbitSequence(x=1, normalized=normalize),
            ta.SaltPepperNoise(p=0.01, normalized=normalize),
        ])
        
        if test:
            self.both_transforms = both_test_transforms
            self.noise_transforms = noise_test_transforms
        else:
            self.both_transforms = both_train_transforms
            self.noise_transforms = noise_train_transforms

        self.start = start
        self.end = end
        self.len = end-start
        if debug:
            print("loading", self.len, "images from", self.start, "to", self.end)
        
        self.videos = np.zeros(dtype=probe_object.dtype, shape=(self.len,) + probe_object.shape)

        for i in range(start, end):
            if debug and i % (self.len//10) == 0:
                print("h5py to numpy:", "{:>4.1f}%".format((i-start)/self.len*100))
            tmp_vid = gnt[i][0]
            self.videos[i-start] = f[tmp_vid]

    def __len__(self):
        'Denotes the total number of samples'
        return self.len
        
    def __getitem__(self, index):
        'Generates one sample of data'
        # currently: stored gt as numpy uint8
        with torch.no_grad():
            gt = self.both_transforms(self.videos[index])
            noise = self.noise_transforms(gt)
        return noise, gt
    