from typing import Optional
import cv2
import numpy as np
import torch
import os
from imageio import imwrite
from pydantic import validator

from torchvision import transforms

from my.utils import (
    tqdm, EventStorage, HeartBeat, EarlyLoopBreak)
from my.config import BaseConf, dispatch, write_full_config
from my.utils.seed import seed_everything

from run_img_sampling import SD, StableDiffusion
from misc import torch_samps_to_imgs
from pose import PoseConfig, camera_pose, sample_near_eye

from run_nerf import VoxConfig
import voxnerf
from voxnerf.utils import every
from my3d import depth_smooth_loss

import run_zero123

device_glb = torch.device("cuda")


class InstructSJC(BaseConf):
    family:     str = "sd"
    sd:         SD = SD(
        variant="instruct_pix2pix",
        text_cfg_scale=7.5,
        image_cfg_scale=1.5,
        scale=None,
        prompt="Turn it into lego blocks.",
        im_path="instruct_3d_to_3d_output/building_lego_planar"
    )
    training_data_dir: str='views_whole_sphere/building_planar'
    lr:         float = 0.05
    n_steps:    int = 10000
    vox:        VoxConfig = VoxConfig(
        model_type="V_SD", grid_size=100, density_shift=-1.0, c=4,
        blend_bg_texture=False, bg_texture_hw=4,
        bbox_len=1.0
    )
    # vox_warmstart_ckpt: Optional[str] = None
    vox_warmstart_ckpt: Optional[str] = 'nerf_output/building_v3_txt_scale_7.5_img_scale_1.5/ckpt/step_9999.pt'
    pose:       PoseConfig = PoseConfig(rend_hw=32, FoV=49.1, R=2.0, test_view_type='planar')

    # Disable depth_smooth_weight for now because it takes forever to run
    depth_smooth_weight: float = 0
    near_view_weight: float = 1e1
    view_weight:        float = 0

    emptiness_weight:   int = 0
    emptiness_scale:    int = 10
    emptiness_step:     float = 0.5
    emptiness_multiplier: float = 20.0

    grad_accum: int = 1

    train_view:         bool = True

    var_red:     bool = True

    save_step: int = 100
    evaluate_step: int = 1000
    checkpoint_step: int = 1000

    @validator("vox")
    def check_vox(cls, vox_cfg, values):
        family = values['family']
        if family == "sd":
            vox_cfg.c = 4
        return vox_cfg

    def run(self):
        cfgs = self.dict()

        family = cfgs.pop("family")
        model = getattr(self, family).make()

        cfgs.pop("vox")
        vox = self.vox.make()

        cfgs.pop("pose")
        poser = self.pose.make()

        if self.vox_warmstart_ckpt:
            state = torch.load(self.vox_warmstart_ckpt, map_location="cpu")
            vox.load_state_dict(state)

        os.makedirs(model.im_path, exist_ok=True)
        write_full_config(self, fname=os.path.join(model.im_path, 'full_config.yaml'))

        instruct_sjc_3d(**cfgs, poser=poser, model=model, vox=vox)


def get_input_fidelity_loss(y, input_image, model):
    y = model.decode(y)
    rgb_loss = ((y - input_image) ** 2).mean()

    input_loss = rgb_loss
    return input_loss

    
def get_nearby_view_loss(y, depth, input_pose, poser, vox, aabb, H, W, K):
    """A regularization to minimize image variation from small change in camera pose"""
    # This is a bit confusing, but we are assuming that the object is
    # in the origin. So we are creating a new pose that starts at
    # near_eye, looking into the -near_eye direction, which goes
    # through the origin.
    # Camera location
    eye = input_pose[:3, -1]
    near_eye = sample_near_eye(eye)
    # Look into the origin
    near_pose = camera_pose(near_eye, -near_eye, poser.up)
    y_near, depth_near, ws_near = run_zero123.render_one_view(vox, aabb, H, W, K, near_pose, return_w=True)
    near_loss = ((y_near - y).abs().mean() + (depth_near - depth).abs().mean())
    return near_loss

    
def get_emptiness_loss(ws, emptiness_scale):
    emptiness_loss = (torch.log(1 + emptiness_scale * ws) * (-1 / 2 * ws)).mean()
    return emptiness_loss


def instruct_sjc_3d(poser, vox, model: StableDiffusion,
    lr, n_steps, emptiness_scale, emptiness_weight, emptiness_step,
    emptiness_multiplier, var_red, train_view, view_weight,
    depth_smooth_weight, near_view_weight, grad_accum, training_data_dir,
    save_step, evaluate_step, checkpoint_step, **kwargs):

    assert model.samps_centered()
    _, target_H, target_W = model.data_shape()
    bs = 1
    aabb = vox.aabb.T.cpu().numpy()
    vox = vox.to(device_glb)
    opt = torch.optim.Adamax(vox.opt_params(), lr=lr)

    H, W = poser.H, poser.W

    ts = model.us[30:-10]
    fuse = EarlyLoopBreak(1)

    folder_name = model.im_path

    # load nerf view
    metadata = voxnerf.data.load_metadata(training_data_dir)
    epoch_size = len(metadata['frames'])
    K = poser.K

    opt.zero_grad()

    with tqdm(total=n_steps) as pbar, \
        HeartBeat(pbar) as hbeat, \
            EventStorage(folder_name) as metric:
        
        tforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop((256, 256))
        ])

        for i in range(n_steps):
            if fuse.on_break():
                break

            input_image, input_pose = voxnerf.data.load_data(root=training_data_dir, step=i%epoch_size, meta=metadata)
            input_image = cv2.resize(input_image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

            # to torch tensor
            with torch.no_grad():
                input_image = torch.as_tensor(input_image, dtype=float, device=device_glb).float()
                input_image = input_image.permute(2, 0, 1)[None, :, :]
                input_image = input_image * 2. - 1.
                input_image = tforms(input_image)
            
            # y: [1, 4, 64, 64] depth: [64, 64]  ws: [n, 4096]
            y, depth, ws = run_zero123.render_one_view(vox, aabb, H, W, K, input_pose, return_w=True)

            # Input regularization
            if train_view and view_weight:
                input_loss =  float(view_weight) * get_input_fidelity_loss(y, input_image, model)
                input_loss.backward(retain_graph=True)

            # near-by view regularziation
            if near_view_weight:
                near_loss = near_view_weight * get_nearby_view_loss(y, depth, input_pose, poser, vox, aabb, H, W, K)
                near_loss.backward(retain_graph=True)

            # depth smoothness loss
            if depth_smooth_weight:
                smooth_loss =  depth_smooth_weight * depth_smooth_loss(depth)
                smooth_loss.backward(retain_graph=True)

            # negative emptiness loss
            if emptiness_weight and i >= emptiness_step * n_steps:
                emptiness_loss = emptiness_weight * get_emptiness_loss(ws, emptiness_scale)
                emptiness_loss = emptiness_loss * (1. + emptiness_multiplier * i / n_steps)
                emptiness_loss.backward(retain_graph=True)

            if isinstance(model, StableDiffusion):
                pass
            else:
                y = torch.nn.functional.interpolate(y, (target_H, target_W), mode='bilinear')

            with torch.no_grad():
                chosen_σs = np.random.choice(ts, bs, replace=False)
                chosen_σs = chosen_σs.reshape(-1, 1, 1, 1)
                chosen_σs = torch.as_tensor(chosen_σs, device=model.device, dtype=torch.float32)

                noise = torch.randn(bs, *y.shape[1:], device=model.device)

                zs = y + chosen_σs * noise

                cond = model.instruct_pix2pix_emb(input_image)
                Ds = model.denoise_instruct_pix2pix(zs, chosen_σs, cond)

                if every(pbar, step=save_step):
                    metric.put_artifact("denoised_image", ".png", lambda fn: imwrite(fn, torch_samps_to_imgs(model.decode(Ds))[0]))

                if var_red:
                    grad = (Ds - y) / chosen_σs
                else:
                    grad = (Ds - zs) / chosen_σs

                grad = grad.mean(0, keepdim=True)

            y.backward(-grad, retain_graph=True)


            if i % grad_accum == (grad_accum-1):
                opt.step()
                opt.zero_grad()

            metric.put_scalars(**run_zero123.tsr_stats(y))

            if every(pbar, step=evaluate_step) and i != 0:
                with EventStorage('eval'):
                    run_zero123.evaluate(model, vox, poser)

            if every(pbar, step=save_step):
                with torch.no_grad():
                    metric.put_artifact("src_view", ".png", lambda fn: imwrite(fn, torch_samps_to_imgs(input_image)[0]))
                    depth_value = depth.clone()
                    if isinstance(model, StableDiffusion):
                        y = model.decode(y)
                    run_zero123.vis_routine(metric, y, depth_value)

            if every(pbar, step=checkpoint_step) and i != 0:
                metric.put_artifact(
                    "ckpt", ".pt", lambda fn: torch.save(vox.state_dict(), fn)
                )

            metric.step()
            pbar.update()
            pbar.set_description(model.im_path)
            hbeat.beat()

        with EventStorage("test"):
            run_zero123.evaluate(model, vox, poser)

        metric.step()

        hbeat.done()


def main():
    seed_everything(0)
    dispatch(InstructSJC, cfg_name="instruct_config.yml")


if __name__ == "__main__":
    main()
