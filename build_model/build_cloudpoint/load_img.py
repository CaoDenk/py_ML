import cv2

"""
    no 数据集编号
"""
def get_img_and_depth(dir_path,dataset_no,imgdir_no,img_name):
    img_fullpath=rf"{dir_path}\rectified{dataset_no}\image0{imgdir_no}\{img_name}"
    
    depth_fullpath=rf"{dir_path}\rectified{dataset_no}\depth0{imgdir_no}\{img_name[:-3]}png"
  
    depth=cv2.imread(depth_fullpath,cv2.IMREAD_UNCHANGED)
    img_bgr=cv2.imread(img_fullpath,cv2.IMREAD_UNCHANGED)
    return img_bgr,depth