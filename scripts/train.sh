## stage1
export WORK_DIR="./job_data_12_21_stage_1_finetune/work_dirs/"
export GPUS=1
export CONFIG="projects/configs/sparsedrive_configs/sparsedrive_small_stage1_perception_fixed.py"
# 可以debug版
# CUDA_VISIBLE_DEVICES=7 python3 -m debugpy --listen 55969 --wait-for-client \


# CUDA_VISIBLE_DEVICES=2,3,5,7 python3 -m debugpy --listen 55969 --wait-for-client \
# CUDA_VISIBLE_DEVICES=6,7 python3 \
# CUDA_VISIBLE_DEVICES=7 python3 -m debugpy --listen 55969 --wait-for-client \
CUDA_VISIBLE_DEVICES=5 python3 \
  -m torch.distributed.run \
  --nproc_per_node=1 \
  --master_port=2334 \
  tools_diffusiondrive/train.py ${CONFIG} \
  --launcher pytorch \
  --deterministic \
  --work-dir ${WORK_DIR}



# export WORK_DIR="./job_data_12_20_stage_1_finetune/work_dirs/"


# bash ./tools/dist_train.sh \
#    projects/configs/sparsedrive_small_stage1_perception_fixed.py \
#    1 \
#    --deterministic

# ## stage2
# bash ./tools/dist_train.sh \
#    projects/configs/sparsedrive_small_stage2.py \
#    8 \
#    --deterministic