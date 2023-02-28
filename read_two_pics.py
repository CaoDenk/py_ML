import cv2
from py_ML.utils.myutils import mat_to_tensor
pic_dir:str = r""
pic_left:str =r"E:\Dataset\dataset\rectified08-003\rectified08\image01\0000000000.jpg"
pic_right:str =r"E:\Dataset\dataset\rectified08-003\rectified08\image02\0000000000.jpg"

mat_left = cv2.imread(pic_left)
mat_right = cv2.imread(pic_right)

tensor_left = mat_to_tensor(mat_left)
tensor_right = mat_to_tensor(mat_right)

print(tensor_left.shape)
