export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH
# python -m debugpy --listen 55969 --wait-for-client \
python -m debugpy --listen 55969 --wait-for-client tools/visualization/visualize.py \
	/data3/zhoujiawei/DiffusionDrive_origin/projects/configs/diffusiondrive_configs/diffusiondrive_small_stage2.py \
	--result-path /data3/zhoujiawei/DiffusionDrive_origin/work_dirs/diffusiondrive_small_stage2_output_not_normalize_1000epoch/results.pkl \
	--out-dir ./vis_output_test_12_23_fix_some_bug_test_last_no_boundary_add_shoulder\