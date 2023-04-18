import cv2

depth = cv2.imread(r'E:\Dataset\OneDrive_1_2022-11-18\rectified22\depth01\0000000000.png', cv2.IMREAD_UNCHANGED)
K = [[417.9036255, 0, 373.208288192749], [0,417.9036255, 158.1358108520508], [0, 0, 1]]
D = [0, 0, 0, 0, 0]  # 相机畸变参数

rgbd_normals = cv2.rgbd.RgbdNormals(depth.shape[0], depth.shape[1], cv2.CV_32F, K, D)
normals = rgbd_normals.compute(depth, False)  # 计算深度图像的法向量

cv2.imshow('normals', normals)
cv2.waitKey()
cv2.destroyAllWindows()
