export WORK_DIR="./job_data_12_21_output_not_normalize_48batchsize_new_pkl/work_dirs/"
export GPUS=2
export CONFIG="./projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py"
# 可以debug版
# CUDA_VISIBLE_DEVICES=7 python3 -m debugpy --listen 55969 --wait-for-client \


# CUDA_VISIBLE_DEVICES=2,3,5,7 python3 -m debugpy --listen 55969 --wait-for-client \
# CUDA_VISIBLE_DEVICES=6,7 python3 \
# CUDA_VISIBLE_DEVICES=7 python3 -m debugpy --listen 55969 --wait-for-client \
CUDA_VISIBLE_DEVICES=2,3 python3 \
  -m torch.distributed.run \
  --nproc_per_node=2 \
  --master_port=2333 \
  tools_diffusiondrive/train.py ${CONFIG} \
  --launcher pytorch \
  --deterministic \
  --work-dir ${WORK_DIR}