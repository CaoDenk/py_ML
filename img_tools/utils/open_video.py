import cv2
file=r"E:\2019\dataset_1\keyframe_2\data\rgb.mp4"
cap=cv2.VideoCapture(file)
i=0
while True:
    ret,frame=cap.read()
    if not ret:
        break
    cv2.imshow("frame",frame)
    print(frame.shape)
    break
    i+=1
    cv2.waitKey(30)

print(i)



