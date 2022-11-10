import utils.myutils as myutils
import cv2
def get_mat_of_count_pixel(img_path:str)->cv2.Mat:
    graypixel_list=myutils.count_pixel(cv2.imread(img_path))
    return myutils.draw_line(graypixel_list)
