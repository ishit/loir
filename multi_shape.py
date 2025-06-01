import sys
import numpy as np
import torch
from model import LOI
import pydiffvg
from torch.utils.tensorboard import SummaryWriter
from kornia.color import rgb_to_hsv
from utils import resolve_collisions
from config import Config
import lpips
from skimage.metrics import structural_similarity as ssim

pydiffvg.set_use_gpu(torch.cuda.is_available())

def generate_circles(cfg):
    x = torch.linspace(-1.5, 1.5, int(np.sqrt(cfg.n_circles)))
    y = torch.linspace(-1.5, 1.5, int(np.sqrt(cfg.n_circles)))
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    orig_pos = torch.stack([xx.flatten(), yy.flatten()], dim=1)
    orig_pos = orig_pos[:cfg.n_circles].to(cfg.device)
    # orig_pos = torch.randn(cfg.n_circles, 2).to(cfg.device)
    with torch.no_grad():
        orig_pos = resolve_collisions(
            orig_pos,
            collision_threshold=cfg.circle_size * 2,
            canvas_width=cfg.canvas_width
        )
    return orig_pos

def main(cfg):
    lpips_loss = lpips.LPIPS(net='vgg').cuda()
    writer = SummaryWriter(cfg.logdir)

    gen = LOI(cfg.loi.bins, cfg.loi.sigma, cfg.loi.alpha, cfg.loi.beta,
              cfg.loi.kernel_size, cfg.loi.bounds)
    canvas_circle = pydiffvg.Circle(radius=torch.tensor(cfg.canvas_width),
                                    center=torch.tensor([cfg.canvas_width / 2,
                                    cfg.canvas_height / 2]))

    canvas_circle_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([0]),
                                              fill_color=cfg.black)

    orig_pos = generate_circles(cfg)

    rand_shapes = []
    rand_groups = []

    for i in range(cfg.n_circles):
        center = torch.tensor([cfg.canvas_width / 2, cfg.canvas_height / 2]).to(cfg.device) + orig_pos[i] * (cfg.canvas_width / 4)
        center = center.clip(0+10, cfg.canvas_width-10)
        radius = torch.tensor(cfg.circle_size)
        rand_shapes.append(pydiffvg.Circle(radius=radius,
                                           center=center))
        rand_groups.append(pydiffvg.ShapeGroup(shape_ids=torch.tensor([i + 1]),
                                               fill_color=torch.rand(4).to(cfg.device)))
        if not cfg.rand_colors:
            rand_groups[-1].fill_color = cfg.all_colors[i % len(cfg.all_colors)]
        else:
            rand_groups[-1].fill_color = torch.rand(4).to(cfg.device)

        rand_groups[-1].fill_color[-1] = 1.0


    shapes = [canvas_circle] + rand_shapes
    shape_groups = [canvas_circle_group] + rand_groups
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        cfg.canvas_width, cfg.canvas_height, shapes, shape_groups)

    render = pydiffvg.RenderFunction.apply
    img = render(cfg.canvas_width,
                 cfg.canvas_height,
                 32,
                 32,
                 0,
                 None,
                 *scene_args)
    target = img.clone().detach()

    target_loi = gen(target[..., :cfg.loi.channels])
    if cfg.hsv_loss:
        target_hsv = rgb_to_hsv(target[..., :3].permute(2, 0, 1).unsqueeze(0)).squeeze(0).permute(1, 2, 0)
        target_loi_hsv = gen(target_hsv)

    writer.add_image('target', target.permute(2, 0, 1).cpu(), 0)
    pydiffvg.imwrite(target.cpu(), f'{cfg.logdir}/target.png')
    pydiffvg.save_svg(f'{cfg.logdir}/target.svg', cfg.canvas_width, cfg.canvas_height, shapes, shape_groups)

    pos = torch.randn(cfg.n_circles, 2).to(cfg.device) / 2
    pos.requires_grad = True

    colors = torch.rand(cfg.n_circles, 4).to(cfg.device)
    colors.requires_grad = True

    for i in range(cfg.n_circles):
        rand_shapes[i].center = torch.tensor([cfg.canvas_width / 2,
                                              cfg.canvas_height / 2]).to(cfg.device) + \
                                pos[i] * (cfg.canvas_width / 4)
        if cfg.optim_color:
            rand_groups[i].fill_color = colors[i]

            with torch.no_grad():
                rand_groups[i].fill_color.data[-1] = 1.0

    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        cfg.canvas_width, cfg.canvas_height, shapes, shape_groups)
    img = render(cfg.canvas_width,
                 cfg.canvas_height,
                 32,
                 32,
                 0,
                 None,
                 *scene_args)
    pydiffvg.imwrite(img.cpu(), f'{cfg.logdir}/init.png')

    # Optimize for radius & center
    if cfg.optim_color:
        optim = torch.optim.Adam([pos, colors], lr=cfg.lr)
    else:
        optim = torch.optim.Adam([pos], lr=cfg.lr)

    for t in range(cfg.iters):
        optim.zero_grad()
        with torch.no_grad():
            pos.data = pos.data.clip(
                -2 + 2 * cfg.circle_size / (cfg.canvas_width / 4),
                2 - 2 * cfg.circle_size / (cfg.canvas_width / 4)
            )
            if cfg.resolve_collisions:
                with torch.no_grad():
                    pos.data = resolve_collisions(pos.data,
                                                collision_threshold=cfg.circle_size*2,
                                                canvas_width=cfg.canvas_width)
            if cfg.optim_color:
                colors.data = colors.data.clip(0, 1)
                colors.data[:, -1] = 1.0

        for i in range(cfg.n_circles):
            rand_shapes[i].center = torch.tensor([cfg.canvas_width / 2,
                                                  cfg.canvas_height / 2]).to(cfg.device) + \
                                    pos[i] * (cfg.canvas_width / 4)
            if cfg.optim_color:
                rand_groups[i].fill_color = colors[i]

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            cfg.canvas_width, cfg.canvas_height, shapes, shape_groups)
        img = render(cfg.canvas_width,
                     cfg.canvas_height,
                     cfg.spp,
                     cfg.spp,
                     np.random.randint(0, 1000000),
                     None,
                     *scene_args)
                     
        if cfg.loi_loss:
            loss = 0
            if cfg.l2_loss:
                loss = loss + (img - target).pow(2).mean()
            else:
                est_loi = gen(img[..., :cfg.loi.channels])
                if cfg.hsv_loss:
                    img_hsv = rgb_to_hsv(img[..., :3].permute(2, 0, 1).unsqueeze(0)) \
                        .squeeze(0).permute(1, 2, 0)
                    est_loi_hsv = gen(img_hsv)
                    loss = loss + (gen.emd(target_loi_hsv, est_loi_hsv) if not cfg.chi2
                                else gen.chi_square(target_loi_hsv, est_loi_hsv))
                if cfg.chi2:
                    loss = gen.chi_square(target_loi, est_loi)
                else:
                    loss = gen.emd(target_loi, est_loi, normalize=False, l1=True, scale_grad=False)
        elif cfg.pyramid_loss:
            # loss = gen.blur_loss(img, target)
            # loss = gen.ssim(img, target)
            loss = gen.pyramid_loss(img, target)
        elif cfg.ssim_loss:
            loss = gen.ssim(img, target)
        elif cfg.lpips_loss:
            loss = lpips_loss(img[..., :cfg.loi.channels].permute(2, 0, 1).unsqueeze(0),
                            target[..., :cfg.loi.channels].permute(2, 0, 1).unsqueeze(0))

        
        mse = (img - target).pow(2).mean()
        psnr = -10 * torch.log10(mse)
        mae = (orig_pos - pos).abs().mean()
        ssim_ = ssim(img[..., :cfg.loi.channels].detach().cpu().numpy(), target[..., :cfg.loi.channels].detach().cpu().numpy(), channel_axis=2, data_range=1.0)
        print('loss:', loss.item())

        if t % cfg.vis_freq == 0:
            writer.add_scalar('loss', loss.item(), t)
            writer.add_scalar('ssim', ssim_.item(), t)
            writer.add_scalar('mae', mae.item(), t)
            writer.add_scalar('mse', mse.item(), t)
            writer.add_scalar('psnr', psnr.item(), t)
            writer.add_image('image', img[..., :cfg.loi.channels].permute(2, 0, 1).cpu(), t)
            pydiffvg.save_svg(f'{cfg.logdir}/image_{t:03d}.svg', cfg.canvas_width, cfg.canvas_height, shapes, shape_groups)
            # save_img(img, f'{cfg.logdir}/image_{t:03d}.png')
            # save_img(img, f'{cfg.logdir}/debug.png')
        loss.backward()
        optim.step()

if __name__ == '__main__':
    config = Config()
    config.vis_freq = 10
    config.iters = 500
    main(config)