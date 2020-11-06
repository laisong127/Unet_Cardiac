import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt

# # HDF5的写入：
# imgData = np.zeros((2, 4))
# f = h5py.File('HDF5_FILE.h5', 'w')  # 创建一个h5文件，文件指针是f
# f['data'] = imgData  # 将数据写入文件的主键data下面
# f['labels'] = np.array([1, 2, 3, 4, 5])  # 将数据写入文件的主键labels下面
# f.close()  # 关闭文件
#
# # HDF5的读取：
f = h5py.File('/home/laisong/github/processed_acdc_dataset/hdf5_files/test_upload/P_138_ES_08_.hdf5', 'r')  # 打开h5文件
# 可以查看所有的主键
for key in f.keys():
    print(f[key].name)
    print(f[key].shape)
    print(f[key][()])



# img = f['image'][()]
# plt.imshow(img,'gray')
# plt.savefig('original.png')
# label = f['label'][()]
# roi_center = f['roi_center'][()]
# roi_radii = f['roi_radii'][()]
# img_ROI = img[(roi_center[0]-2*roi_radii[0]):(roi_center[0]+2*roi_radii[0]),(roi_center[1]-2*roi_radii[1]):(roi_center[1]+2*roi_radii[1])]
# # img_ROI = img[(roi_center[0]-64):(roi_center[0]+64),(roi_center[1]-64):(roi_center[1]+64)]
# plt.imshow(img_ROI,'gray')
# plt.savefig('img_ROI.png')
# label_ROI = label[(roi_center[0]-2*roi_radii[0]):(roi_center[0]+2*roi_radii[0]),(roi_center[1]-2*roi_radii[1]):(roi_center[1]+2*roi_radii[1])]
# plt.imshow(label,'gray')
# plt.savefig('label.png')
# plt.imshow(label_ROI,'gray')
# plt.savefig('label_ROI.png')