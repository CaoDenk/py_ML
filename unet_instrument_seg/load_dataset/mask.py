import cv2
import numpy as np





if __name__ =='__main__':
    img = cv2.imread('image.png')

    # 创建掩膜
    mask = np.zeros(img.shape[:2], np.uint8)
    pts = np.array([[10, 50], [400, 50], [400, 200], [10, 200]], np.int32)
    cv2.fillPoly(mask, [pts], (255, 255, 255))

    # 应用掩膜
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    # 显示结果
    cv2.imshow("Original Image", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Masked Image", masked_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()