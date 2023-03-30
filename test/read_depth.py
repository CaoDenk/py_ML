import cv2

file=r"E:\Dataset\dataset\rectified08-003\rectified08\depth01\0000000000.png"



m=cv2.imread(file)
h,w,c =m.shape

for i in range(h//10):
    for j in range(w//10):
        for k in range(c):
            print(m[i,j,k],end=',')



