import json
import os

import numpy as np
import torch

from adapt import ScoreAdapter
from imageio import imwrite
from misc import torch_samps_to_imgs
from my.config import optional_load_config
from my.utils import  tqdm, EventStorage, EarlyLoopBreak, get_event_storage
from my.utils.seed import seed_everything
from run_img_sampling import SD, StableDiffusion
from run_zero123 import SJC, render_one_view
from voxnerf.vis import bad_vis as nerf_vis

device_glb = torch.device("cuda")


def GenerateTrainingDataFromZero123():
    # Point these to the config and checkpoints of a nerf
    zero123_nerf_ckpt = 'experiments/exp_wild/scene-spyro-index-0_scale-100.0_train-view-True_view-weight-10000_depth-smooth-wt-10000.0_near-view-wt-10000.0/ckpt/step_10000.pt'
    cfg = optional_load_config(fname="zero123_config.yml")

    num_training_points = 1000

    assert len(cfg) > 0, "can't find cfg file"
    mod = SJC(**cfg)

    family = cfg.pop("family")
    model: ScoreAdapter = getattr(mod, family).make()
    vox = mod.vox.make()
    poser = mod.pose.make()

    with EventStorage('zero123_nerf_output'):
        state = torch.load(zero123_nerf_ckpt, map_location="cpu")
        vox.load_state_dict(state)
        vox.to(device_glb)

        GenerateTrainingPoints(model, vox, poser, num_training_points)


@torch.no_grad()
def GenerateTrainingPoints(score_model, vox, poser, num_points):
    H, W = poser.H, poser.W
    vox.eval()
    Ks, poses, _ = poser.sample_train(num_points)
    K = Ks[0]
    fov = K[0][0]
    camera_angle_x = 2 * np.arctan(W / (2 * fov))

    transform_data = {}
    transform_data['camera_angle_x'] = camera_angle_x
    transform_data['frames'] = []

    fuse = EarlyLoopBreak(5)
    metric = get_event_storage()

    aabb = vox.aabb.T.cpu().numpy()
    vox = vox.to(device_glb)

    for idx in tqdm(range(len(poses))):
        if fuse.on_break():
            break

        pose = poses[idx]

        y, depth = render_one_view(vox, aabb, H, W, K, pose)
        if isinstance(score_model, StableDiffusion):
            y = score_model.decode(y)
        pane = nerf_vis(y, depth, final_H=256)
        im = torch_samps_to_imgs(y)[0]
        metric.put_artifact("train", ".png", lambda fn: imwrite(fn, im))
        transform_data['frames'].append({'file_path': f'./train/step_{idx}.png',
                                           'transform_matrix': pose.tolist()})

        metric.step()

    metric.flush_history()

    with open(os.path.join(metric.output_dir, 'transform_train.json'), 'w') as f:
        json.dump(transform_data, f, indent=2)


def ValidateTrainingDataFromZero123():
    # Point these to the config and checkpoints of a nerf
    zero123_nerf_ckpt = 'experiments/exp_wild/scene-spyro-index-0_scale-100.0_train-view-True_view-weight-10000_depth-smooth-wt-10000.0_near-view-wt-10000.0/ckpt/step_10000.pt'
    cfg = optional_load_config(fname="zero123_config.yml")


    assert len(cfg) > 0, "can't find cfg file"
    mod = SJC(**cfg)

    family = cfg.pop("family")
    model: ScoreAdapter = getattr(mod, family).make()
    vox = mod.vox.make()
    poser = mod.pose.make()

    with EventStorage('zero123_nerf_output'):
        state = torch.load(zero123_nerf_ckpt, map_location="cpu")
        vox.load_state_dict(state)
        vox.to(device_glb)

        ValidateTrainingPoints(model, vox, poser)


@torch.no_grad()
def ValidateTrainingPoints(score_model, vox, poser):
    H, W, K = poser.H, poser.W, poser.K
    vox.eval()

    fuse = EarlyLoopBreak(5)
    metric = get_event_storage()

    with open(os.path.join(metric.output_dir, 'transform_train.json'), 'r') as f:
        transform_data = json.load(f)

    frames = transform_data['frames']
    num_data = len(frames)

    aabb = vox.aabb.T.cpu().numpy()
    vox = vox.to(device_glb)

    for idx in tqdm(range(num_data)):
        if fuse.on_break():
            break

        pose = np.array(frames[idx]['transform_matrix'])

        y, depth = render_one_view(vox, aabb, H, W, K, pose)
        if isinstance(score_model, StableDiffusion):
            y = score_model.decode(y)
        im = torch_samps_to_imgs(y)[0]
        metric.put_artifact("validate", ".png", lambda fn: imwrite(fn, im))

        metric.step()

    metric.flush_history()


def main():
    seed_everything(0)
    GenerateTrainingDataFromZero123()
    print('Validating generated data. Feel free to terminate early.')
    ValidateTrainingDataFromZero123()


if __name__ == "__main__":
    main()
