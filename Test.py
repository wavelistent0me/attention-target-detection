import cv2

cap = cv2.VideoCapture(0)  # 打开默认摄像头

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    ret, frame = cap.read()
    frame=cv2.imread("C:/MineApp/MyProgram/attention-target-detection/data/demo/frames/00002575.jpg")
    if not ret:
        print("读取摄像头失败")
        break

    cv2.imshow('Camera', frame)  # 显示摄像头画面

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
        break

cap.release()
cv2.destroyAllWindows()