import cv2
import os


def create_dir(l):
    for i in l:
        if not os.path.exists(i):
            os.mkdir(i)

file=r"E:\2019\dataset_1\keyframe_3\data\rgb.mp4"
cap=cv2.VideoCapture(file)

par_dir=os.path.dirname(file)

par_par_dir=os.path.dirname(par_dir)



left_dir=rf"{par_par_dir}\left"
right_dir=rf"{par_par_dir}\right"

create_dir((left_dir,right_dir))



i=0
while True:
    ret,frame=cap.read()
    if not ret:
        break
    # cv2.imshow("frame",frame)
    # height=frame.shape[0]
    height=frame.shape[0]
    spli_h=height//2
    left=frame[:spli_h,...]
    right=frame[spli_h:,...]
    left_path=rf"{left_dir}\{i}.jpg"
    # cv2.imwrite(left_path,left)
    
    right_path=rf"{right_dir}\{i}.jpg"
    # cv2.imwrite(right_dir,right)

    cv2.imwrite(left_path,left)
    cv2.imwrite(right_path,right)
    i+=1


print(i)
