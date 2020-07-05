import torch
import torch.nn as nn
import skimage.measure as skm


'''
Charbonnier Penalty Function (smoothed L1 loss)
'''
class CharbonnierPenalty(nn.Module):
    def __init__(self, n=1e-3, total_variation=False, lam=1e-6, per_pixel=False):
        super().__init__()
        self.n = n
        self.total_variation = total_variation
        self.lam = lam
        self.per_pixel = per_pixel
        
    def forward(self, output, gt):
        assert output.shape == gt.shape, "output and gt shapes do not match"
        
        # batch x colors x frames x height x width 
        x = output.sub(gt)
        loss = torch.sqrt(x*x + self.n*self.n)
        if self.total_variation:
            loss += self.lam*(torch.sum(torch.abs(x[:,:,:,:-1]-x[:,:,:,1:]))
                             +torch.sum(torch.abs(x[:,:,:-1,:]-x[:,:,1:,:]))
                             +torch.sum(torch.abs(x[:,:-1,:,:]-x[:,1:,:,:])))
        loss = loss.mean() if self.per_pixel else loss.sum()/output.shape[0]
        return loss
    
    def __repr__(self):
        lmbda = "" if not self.total_variation else ", lambda="+str(self.lam)
        return "{}_v3(n={}, total_variation={}".format(self.__class__.__name__, self.n, self.total_variation)+lmbda+', per_pixel='+str(self.per_pixel)+')'


'''
Calculate SSIM and PSNR
'''
def calculate_ssim(output, gt):
    ssim = skm.compare_ssim(X=output, Y=gt, data_range=(output.max()-output.min()), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    return ssim

def calculate_psnr(output, gt):
    psnr = skm.compare_psnr(im_true=gt, im_test=output)
    return psnr