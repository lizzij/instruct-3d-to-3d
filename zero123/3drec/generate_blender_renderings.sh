OBJECT_PATH="/home/chi/.objaverse/hf-objaverse-v1/glbs/000-023/8ff7f1f2465347cd8b80c9b206c2781e.glb" # Any .glb file from objaverse should work
OUTPUT_DIR="views_whole_sphere"
NUM_IMAGES=100
SAMPLE_TYPE="spherical"
CUDA_VISIBLE_DEVICES=0  ../objaverse-rendering/blender-3.2.2-linux-x64/blender \
  -b -P ../objaverse-rendering/scripts/generate_blender_renderings.py -- \
  --object_path  "$OBJECT_PATH" --output_dir "$OUTPUT_DIR" \
  --num_images $NUM_IMAGES --sample_type "$SAMPLE_TYPE"