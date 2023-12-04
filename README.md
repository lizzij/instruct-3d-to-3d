# Instruct 3D-to-3D
CS236 project

## Set up zero-123
```
pushd zero123
conda create -n zero123 python=3.9
conda activate zero123
cd zero123
pip install -r requirements.txt
git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/
git clone https://github.com/openai/CLIP.git
pip install -e CLIP/

wget https://cv.cs.columbia.edu/zero123/assets/105000.ckpt
popd
```

### Set up objaverse rendering

[!CAUTION]
Downloading the whole objaverse rendering takes more than 1.5TB. Do not download it unless you are certain that your machine can handle it.
If your machine doesn't have enough disk space, follow the instruction below to manually render the 3D models.

To download the whole objaverse renderings, run
```wget https://tri-ml-public.s3.amazonaws.com/datasets/views_release.tar.gz```

Otherwise, do the following:
```
pushd objaverse-rendering
bash setup.sh
# Download 100 3D-models
python scripts/download.py
# Render them
python scripts/distributed.py \
	--num_gpus 1 \
	--workers_per_gpu 2 \
	--input_models_path input_models_path.json
popd
```

### Set up 3D reconstruction
```
pushd 3drec
pip install -r requirements.txt
popd
```

### Do 3d reconstruction
This will look up the image in `3drec/experiments/nerf_wild/$scene/` and try to do 3d reconstruction based on that.
The output is stored in `3drec/data/nerf_wild/$args/`.
```
cd 3drec
pip install -r requirements.txt
python run_zero123.py \
    --scene pikachu \
    --index 0 \
    --n_steps 10000 \
    --lr 0.05 \
    --sd.scale 100.0 \
    --emptiness_weight 0 \
    --depth_smooth_weight 10000. \
    --near_view_weight 10000. \
    --train_view True \
    --prefix "experiments/exp_wild" \
    --vox.blend_bg_texture False \
    --nerf_path "data/nerf_wild"
```

## Instruct-Pix2Pix
The above set up should get Instruct-Pix2Pix as well.
To test this, run the following command:
```
python edit_cli.py --steps 100 --resolution 512 --seed 1371 --cfg-text 7.5 \
    --cfg-image 1.2 --input imgs/example.jpg --output imgs/output.jpg --edit \
     "turn him into a cyborg"
```