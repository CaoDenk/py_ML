import utils.myutils as myutils
import cv2
def get_histogram_from_str(img_path:str)->cv2.Mat:
    graypixel_list=myutils.count_pixel(cv2.imread(img_path))
    return myutils.to_histogram(graypixel_list)
def get_histogram_from_mat(img_mat:cv2.Mat)->cv2.Mat:
    graypixel_list=myutils.count_pixel(img_mat)
    return myutils.to_histogram(graypixel_list)