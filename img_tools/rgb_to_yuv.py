import cv2
import numpy as np






if __name__ =='__main__':
        file=r"E:\Dataset\dataset\mesad-real\mesad-real\train\images\real1_frame_490.jpg"
        # cv2.rgb_to_hsv(r, g, b)
        # cv2.rgb_
        # cv2.rgb_to
        mat =cv2.imread(file)
        hls_mat=cv2.cvtColor(mat,cv2.COLOR_BGR2HLS)
        hsv_mat=cv2.cvtColor(mat,cv2.COLOR_BGR2HSV)
        # print(hsv_mat.shape)
        # h,w,c=hsv_mat.shape
        # for i in range(h):
        #     for j in range(w):
        #         # for k in range(c):
        #         hsv_mat[i,j,1]=0
        #         hsv_mat[i,j,1]=0
                
        # # hsv_mat[::]
        # cv2.imshow("hsv",hsv_mat)
        # cv2.waitKey()

        mask = cv2.inRange(hls_mat, np.array([0,250,0]), np.array([255,255,255]))

        # Apply Mask to original image
        white_mask = cv2.bitwise_and(mat, mat, mask=mask)

        # h,w,c=hls_mat.shape
        # for i in range(h):
        #     for j in range(w):
        #         # for k in range(c):
        #         hls_mat[i,j,2]=0
                # hls_mat[i,j,1]=0
        cv2.imshow("hsv",white_mask)
        cv2.waitKey()