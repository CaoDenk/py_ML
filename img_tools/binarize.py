import cv2
import numpy as np



def bin(file,threshold):
        mat =cv2.imread(file)

        cv2.cvtColor(mat,cv2.COLOR_BGR2GRAY,mat)

        mat=cv2.threshold(mat,threshold,255)


if __name__ =='__main__':
        file=r"C:\Users\denk\Pictures\微信图片_20230415115929.jpg"
        color_image = cv2.imread(file)

# 将彩色图像转换为灰度图像
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray",gray_image)
        
        h,w=gray_image.shape
        # for i in range(h):
        #         for j in range(w):
        #                 print(gray_image[i,j],end=",")
        #         print()
        # cv2.waitKey()
        # exit()
        # 二值化图像
        thresh_value =30
        max_value = 255
        thresh_type = cv2.THRESH_BINARY
        retval, binary_image = cv2.threshold(gray_image, thresh_value, max_value, thresh_type)

        # 显示原图、灰度图和二值图
        # cv2.imshow('Color Image', color_image)
        # cv2.imshow('Gray Image', gray_image)
        cv2.imshow('Binary Image', binary_image)
        cv2.imwrite("white2.png",binary_image)
        cv2.waitKey()
        
