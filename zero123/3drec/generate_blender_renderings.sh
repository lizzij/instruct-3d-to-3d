#OBJECT_PATH="/home/chi/.objaverse/hf-objaverse-v1/glbs/000-023/8ff7f1f2465347cd8b80c9b206c2781e.glb" # Any .glb file from objaverse should work
OBJECT_PATH="/home/chi/.objaverse/hf-objaverse-v1/glbs/000-043/fef8c7a37ebe46efba7efce99e39c3e0.glb"
OUTPUT_DIR="views_whole_sphere"
NUM_IMAGES=1000
SAMPLE_TYPE="planar"
OUTPUT_NAME="building_planar"
CUDA_VISIBLE_DEVICES=0  ../objaverse-rendering/blender-3.2.2-linux-x64/blender \
  -b -P ../objaverse-rendering/scripts/generate_blender_renderings.py -- \
  --object_path  "$OBJECT_PATH" --output_dir "$OUTPUT_DIR" \
  --num_images "$NUM_IMAGES" --sample_type "$SAMPLE_TYPE" \
  --output_name "$OUTPUT_NAME"
