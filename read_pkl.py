import pickle
import sys
sys.path.append('/data3/zhoujiawei/DiffusionDrive_origin')
with open('/data3/zhoujiawei/DiffusionDrive_origin/work_dirs/sparsedrive_small_stage1_perception_fixed_base_sparsedrive_stage1/results.pkl', 'rb') as f:
    data = pickle.load(f)
print(data.keys())