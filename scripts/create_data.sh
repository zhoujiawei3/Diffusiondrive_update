export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

# python -m debugpy --listen 55969 --wait-for-client tools/data_converter/nuscenes_converter.py nuscenes \

# python tools/data_converter/nuscenes_converter.py nuscenes \
#     --root-path ./data/nuscenes \
#     --canbus ./data/nuscenes \
#     --out-dir ./data/infos/ \
#     --extra-tag nuscenes \
#     --version v1.0-mini

python tools_diffusiondrive/data_converter/nuscenes_converter_carla_ego.py nuscenes \
    --root-path ./data/nuscenes \
    --canbus ./data/nuscenes \
    --out-dir ./data/infos/ \
    --extra-tag nuscenes \
    --version v1.0 \
    --workers 80

