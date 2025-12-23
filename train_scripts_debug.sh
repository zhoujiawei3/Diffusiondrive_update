export WORK_DIR="./job_data_12_20_output_normalize/work_dirs/"
export GPUS=4
export CONFIG="./projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2_multi_car.py"
# 可以debug版
# CUDA_VISIBLE_DEVICES=7 python3 -m debugpy --listen 55969 --wait-for-client \


# CUDA_VISIBLE_DEVICES=2,3,5,7 python3 -m debugpy --listen 55969 --wait-for-client \
# CUDA_VISIBLE_DEVICES=6,7 python3 \
# CUDA_VISIBLE_DEVICES=7 python3 -m debugpy --listen 55969 --wait-for-client \
# CUDA_VISIBLE_DEVICES=4 python3 \
CUDA_VISIBLE_DEVICES=4 python3 -m debugpy --listen 55969 --wait-for-client \
  -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port=2335 \
  tools_diffusiondrive/train.py ${CONFIG} \
  --launcher pytorch \
  --deterministic \
  --work-dir ${WORK_DIR}