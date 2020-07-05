import torch


class OneCycleScheduler(object):
    def __init__(self, epochs=400, hi_lr=1e-3, low_lr=1e-4, tail_epochs=50, tail_multiplier=0.1):
        self.body = epochs-tail_epochs
        self.peak = self.body/2
        self.ratio = low_lr/hi_lr
        self.ratio_tail = low_lr*tail_multiplier
        self.tail_epochs = tail_epochs
        self.hi_lr = hi_lr
        self.low_lr = low_lr
        self.tail_multiplier = tail_multiplier
        
    def __call__(self, e):
        frac = (e+1)/self.body
        if e < self.peak:
            return lerp(self.ratio, 2-self.ratio, frac)
        if e < self.body:
            return lerp(2-self.ratio, self.ratio, frac)
        frac = (e-self.body)/self.tail_epochs
        return lerp(self.ratio, self.ratio_tail, frac)
    
    def __repr__(self):
        return "{}(hi_lr={}, low_lr={}, tail_epochs={}, tail_multiplier={})".format(
            self.__class__.__name__, self.hi_lr, self.low_lr, self.tail_epochs, self.tail_multiplier)
    
lerp = lambda a,b,t: (1-t)*a + t*b
