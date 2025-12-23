import numpy as np
data_path="/data3/zhoujiawei/DiffusionDrive_origin/data/kmeans/kmeans_motion_6.npy"
data = np.load(data_path)
print(data.shape)
print(data)



#找到x，y相对前一步增量的最大值和最小值
x = data[:,:,:,0]
y = data[:,:,:,1]
#增加一个0在最前面，表示第一步的增量为0
x = np.concatenate((np.zeros((x.shape[0],x.shape[1],1)),x),axis=2)
y = np.concatenate((np.zeros((y.shape[0],y.shape[1],1)),y),axis=2)
x_diff = x[:, :, 1:] - x[:, :, :-1]
y_diff = y[:, :, 1:] - y[:, :, :-1]
print("x_diff max:", np.max(x_diff))
print("x_diff min:", np.min(x_diff))
print("y_diff max:", np.max(y_diff))
print("y_diff min:", np.min(y_diff))


