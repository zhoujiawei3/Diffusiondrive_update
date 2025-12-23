bash ./tools/dist_test.sh \
    /data3/zhoujiawei/DiffusionDrive_origin/projects/configs/sparsedrive_configs/sparsedrive_small_stage1_perception_fixed.py \
    /data3/zhoujiawei/DiffusionDrive_origin/job_data_12_21_stage_1_finetune/work_dirs/iter_1000.pth \
    1 \
    --deterministic \
    --eval bbox
    # --result_file ./work_dirs/sparsedrive_small_stage2/results.pkl