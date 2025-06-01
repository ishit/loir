import torch
import torch.nn as nn
import drjit as dr
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import GaussianBlur
from PIL import Image
import pytorch_msssim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def to_np(tensor):
    return tensor.detach().cpu().numpy()

torch.Tensor.to_np = to_np

def gaussian_kernel(size: int, sigma: float):
    x = torch.linspace(-size // 2 + 1, size // 2, size)
    y = torch.linspace(-size // 2 + 1, size // 2, size)
    x_grid, y_grid = torch.meshgrid(x, y)
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    # kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2)) * 0 + 1
    kernel = kernel / torch.sum(kernel)
    return kernel

class LOI(nn.Module):
    def __init__(self, n_bins, sigma, alpha, beta, kernel_size, bounds,
    channels=3):
        super(LOI, self).__init__()
        self.n_bins = n_bins
        self.sigma = sigma
        self.alpha = alpha
        self.kernel_size = kernel_size
        self.bounds = bounds
        self.beta = beta
        self.sigma_kernel_size = max(3, int(sigma[-1] * 4))
        if self.sigma_kernel_size % 2 == 0:
            self.sigma_kernel_size += 1
        self.blur_kernels = [GaussianBlur(kernel_size=kernel_size, sigma=s) for s in alpha]
        self.blur = GaussianBlur(kernel_size=5, sigma=1)
        
        self.sigma_kernel = [gaussian_kernel(self.sigma_kernel_size, s) for s in sigma]
        self.sigma_kernel = torch.stack(self.sigma_kernel, dim=0).unsqueeze(1).repeat(channels, 1, 1, 1).to(device)  # [n_sigma, 1, kernel_size, kernel_size]
        self.bin_val = torch.linspace(bounds[0], bounds[1], n_bins[-1] + 1).to(device) # with max n_bins
        self.bin_val = (self.bin_val[1:] + self.bin_val[:-1]) / 2
        self.bin_val = self.bin_val.view(1, n_bins[-1], 1, 1, 1, 1).expand(len(beta), -1, 1, 1, 1, 1) # [beta, bins, 1, 1, 1, 1]
        self.beta_val = torch.tensor(self.beta).to(device).view(-1, 1, 1, 1, 1, 1).expand(-1, n_bins[-1], 1, 1, 1, 1) # [beta, bins, 1, 1, 1, 1]
        self.alpha_kernel = [gaussian_kernel(kernel_size, s) for s in alpha]
        self.alpha_kernel = torch.stack(self.alpha_kernel, dim=0).unsqueeze(1).to(device)  # [alpha, 1, kernel_size, kernel_size]
        self.n_thetas = 1
        self.alpha_kernel = self.alpha_kernel.repeat(len(self.beta) * self.n_bins[-1] * channels * len(self.sigma), 1, 1, 1)

    def forward(self, im_th): # expected input: [H, W, C]
        im_th = im_th.permute(2, 0, 1)
        with torch.no_grad():
            im_th.data = im_th.data.clamp(0, 1)
        C, H, W = im_th.shape

        # Convolve the image with different sigma values
        im_th_convolved = F.conv2d(
            im_th.unsqueeze(0),
            self.sigma_kernel,
            padding=self.sigma_kernel_size // 2,
            groups=C
        )
        im_th_convolved = im_th_convolved.view(C, -1, H, W) # [C, n_sigma, H, W]

        # Compute isophote images
        im_diff = (im_th_convolved.unsqueeze(0).unsqueeze(1) - self.bin_val)
        im_iso = torch.exp(-0.5 * (im_diff / self.beta_val).pow(2))
        im_iso = im_iso * (1 / (np.sqrt(2 * np.pi) * self.beta_val)) # [beta, bins, C, n_sigma, H, W]
        im_iso = im_iso.reshape(len(self.beta) * self.n_bins[-1] * C * len(self.sigma), H, W)

        # Convolve the isophote images with different alpha values
        loi = F.conv2d(
            im_iso.unsqueeze(0),
            self.alpha_kernel,
            padding=self.kernel_size // 2,
            groups=len(self.beta) * self.n_bins[-1] * C * len(self.sigma)
        ) # [beta * bins * C * n_sigma, alpha, H, W]
        loi = loi.reshape(len(self.beta), self.n_bins[-1], C, len(self.sigma), len(self.alpha)*self.n_thetas, H, W)
        loi = loi.permute(4, 0, 3, 2, 5, 6, 1)
        return loi

    @dr.wrap_ad(source='drjit', target='torch')
    def forward_dr(self, im_th):
        return self.forward(im_th)

    def get_mode(self, loi):
        idx = torch.argmax(loi, dim=-1)
        val = (idx / self.n_bins[-1]) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        return val

    def get_median(self, loi):
        idx = torch.median(loi, dim=-1)
        # val = (idx / self.n_bins[-1]) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        return idx

    def sample_oi(self, loi, a=-1, b=-1, s=-1):
        oi_samples = 1
        C, H, W = loi.shape[3:6]
        oi = torch.multinomial(loi[a, b, s].reshape(-1, self.n_bins[-1])+1e-6, oi_samples, replacement=True) / (self.n_bins[-1] - 1)
        oi = oi.view(C, H, W, oi_samples)
        return oi

    def render_oi(self, loi, path='oi.png', gamma=2.2):
        plt.clf()
        fig, axs = plt.subplots(loi.shape[2], loi.shape[0], figsize=(50, 50))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(loi.shape[0]):
            for j in range(loi.shape[2]):
                oi = self.sample_oi(loi, a=i, s=j)
                gamma_corrected = np.power(oi[..., 0].permute(1, 2, 0).detach().cpu().numpy(), 1.0/gamma)
                self.render_im(gamma_corrected, path=f'../oi_{i}_{j}.png', gamma=1.0)
                if loi.shape[2] == 1:
                    axs[i].imshow(gamma_corrected)
                    axs[i].axis('off')
                else:
                    axs[j][i].imshow(gamma_corrected)
                    axs[j][i].axis('off')
        plt.axis('off')
        plt.tight_layout(pad=0.00)
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def render_im(self, im, path='im.png', gamma=2.2):
        plt.clf()
        if isinstance(im, torch.Tensor):
            im = to_np(im)

        im = np.power(im, 1/gamma)
        plt.imshow(im)
        plt.axis('off')
        plt.gca().set_position([0, 0, 1, 1])
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
        plt.close()

    def emd(self, hist1, hist2, normalize=False, l1=True, scale_grad=False):
        # Ensure the histograms are normalized
        if normalize:
            if scale_grad:
                hist1 = hist1 / (hist1.sum(dim=-1, keepdim=True) + 1e-6)
                hist2 = hist2 / (hist2.sum(dim=-1, keepdim=True) + 1e-6)
            else:
                with torch.no_grad():
                    hist1.data = hist1.data / (hist1.sum(dim=-1, keepdim=True) + 1e-6)
                    hist2.data = hist2.data / (hist2.sum(dim=-1, keepdim=True) + 1e-6)

        # Compute the cumulative distribution functions (CDFs)
        cdf1 = torch.cumsum(hist1, dim=-1)
        cdf2 = torch.cumsum(hist2, dim=-1)

        # Compute the Earth Mover's Distance
        if l1:  
            emd = torch.mean(torch.abs(cdf1 - cdf2))
        else:
            emd = torch.mean((cdf1 - cdf2).pow(2))

        return emd

    @dr.wrap_ad(source='drjit', target='torch')
    def emd_dr(self, hist1, hist2, normalize=True, l1=False):
        return self.emd(hist1, hist2, normalize=normalize, l1=l1, scale_grad=True)

    def pyramid_loss(self, img, target, levels=0):
        img = img.permute(2, 0, 1).unsqueeze(0)
        target = target.permute(2, 0, 1).unsqueeze(0)
        if levels == 0:
            levels = int(np.log2(min(img.shape[2:4]))) - 1

        loss = 0
        for _ in range(levels):
            loss = loss + (self.blur(img) - self.blur(target)).abs().mean()
            img = F.interpolate(img, scale_factor=0.5, mode='nearest')
            target = F.interpolate(target, scale_factor=0.5, mode='nearest')
        return loss
    
    def ssim(self, img, target):
        return pytorch_msssim.ms_ssim(img.permute(2, 0, 1).unsqueeze(0), target.permute(2, 0, 1).unsqueeze(0), data_range=1.0)

    @dr.wrap_ad(source='drjit', target='torch')
    def blur_loss_dr(self, img, target):
        return self.pyramid_loss(img, target)

    @dr.wrap_ad(source='drjit', target='torch')
    def ssim_dr(self, img, target):
        return self.ssim(img, target)