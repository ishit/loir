import torch

class Config:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    canvas_width = 168
    canvas_height = 168
    spp = 16

    class LOIConfig:
        bins = [8]
        sigma = [1, 15, 45]
        alpha = [1, 15, 45]
        beta = [1/bins[0]]
        bounds = [0-4*beta[0], 1+4*beta[0]]
        kernel_size = 121
        channels = 3

    loi = LOIConfig()

    n_circles = 16**2
    circle_size = 3
    logdir = 'results/ours'
    loi_loss = True
    pyramid_loss = False
    ssim_loss = False
    lpips_loss = False
    optim_color = False
    rand_colors = False
    l2_loss = False
    chi2 = False
    hsv_loss = False
    iters = 2000
    vis_freq = 1
    lr = 1e-1
    resolve_collisions = True

    # colors
    black = torch.tensor([0.00, 0.00, 0.0, 1.0]).to(device)
    green = torch.tensor([0.00, 1.00, 0.0, 1.0]).to(device)
    red = torch.tensor([1.00, 0.00, 0.0, 1.0]).to(device)
    blue = torch.tensor([0.00, 0.00, 1.0, 1.0]).to(device)
    white = torch.tensor([1.00, 1.00, 1.0, 1.0]).to(device)
    all_colors = [green, red, blue]
    for c in all_colors:
        c[-1] = 1.0