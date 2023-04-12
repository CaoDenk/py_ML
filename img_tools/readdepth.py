import cv2



def check_depth():
    img=r"E:\Dataset\OneDrive_1_2022-11-18 (1)\rectified04\depth02\0000000000.png"
    mat=cv2.imread(img,cv2.IMREAD_UNCHANGED)

    cv2.imshow("depth",mat*3)
    cv2.waitKey()


    mat.max()

    orig=r"E:\Dataset\OneDrive_1_2022-11-18 (1)\rectified04\image01\0000000000.png"

    print(mat.shape)

    h,w =mat.shape

    for i in range(h):
        for j in range(w):
                print(f"{mat[i,j]}",end=",")


def show_depth(img):
    mat=cv2.imread(img,cv2.IMREAD_UNCHANGED)
    h,w =mat.shape

    for i in range(h):
        for j in range(w):
                print(f"{mat[i,j]}",end=",")
                
if __name__ =='__main__':   
    show_depth(r"E:\ml\PSMNet\Test_disparity.png")