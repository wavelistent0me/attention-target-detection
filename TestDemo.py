from model import ModelSpatial
import torch
import cv2
from torchvision import transforms
# 配置文件,定义了一些变量
from config import *
from utils import imutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 图像转张量方法
# 统一图像尺寸（调整为 input_resolution × input_resolution）
# 转换为张量格式
# 图像标准化（归一化）
def _get_transform():
    transform_list = []
    transform_list.append(transforms.ToPILImage()) # 兼容 OpenCV 图像
    transform_list.append(transforms.Resize((input_resolution, input_resolution)))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transform_list)

test_transforms=_get_transform()

model=ModelSpatial()
model_dict=model.state_dict()
pretrained_dict=torch.load('model_demo.pt')
pretrained_dict=pretrained_dict['model']
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# 加载人脸识别Haar模型
face_cascade = cv2.CascadeClassifier("C:/MineApp/MyProgram/attention-target-detection/haarcascade_frontalface_alt2.xml")
cap=cv2.VideoCapture(1)

with torch.no_grad():
    while True:
            ret,frame=cap.read()
            if not ret:
                break

            # 测试用,加载一张图片
            frame=cv2.imread("C:/MineApp/MyProgram/attention-target-detection/data/demo/frames/00002575.jpg")

            frame_raw=frame.copy()

            height, width = frame.shape[:2]

            # 转为灰度图（Haar 级联需要灰度图）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            # 检测人脸
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            faces = sorted(faces, key=lambda box: box[2] * box[3], reverse=True)
            # x,y是左上角的坐标,w,h是宽和高
            if len(faces)>0:
                x, y, w, h = faces[0]
            else:
                x, y, w, h = 0, 0, 0, 0
            # 左,上,右,下
            head_box=[x, y, x+w, y+h]
            # 标准数据
            # head_box=[533.1, 50.599999999999994, 773.89, 297.44]

            # 裁剪出头部的位置
            head = frame[y:y+h, x:x+w]

            head=test_transforms(head)
            frame=test_transforms(frame_rgb)

            # 将头部框编码成一个与输入图像大小一致的 mask，表示“头的位置”。不知道什么意思
            head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3], width, height,
                                                            resolution=input_resolution).unsqueeze(0)
            head = head.unsqueeze(0).cuda()
            frame = frame.unsqueeze(0).cuda()
            head_channel = head_channel.unsqueeze(0).cuda()

            # 推理,也就是前向传播
            raw_hm, _, inout = model(frame, head_channel, head)

            if False:
                    raw_hm = raw_hm.cpu().detach().numpy() * 255
                    raw_hm = raw_hm.squeeze()
                    inout = inout.cpu().detach().numpy()
                    inout = 1 / (1 + np.exp(-inout))
                    inout = (1 - inout) * 255
                    im_resized = Image.fromarray(raw_hm).resize((width, height), Image.BILINEAR)
                    norm_map = np.array(im_resized) - inout
                    plt.close()
                    fig = plt.figure()
                    fig.canvas.manager.window.move(0,0)
                    plt.axis('off')
                    plt.imshow(norm_map)
                    # plt.imshow(norm_map, cmap = 'jet', alpha=0.2, vmin=0, vmax=255)
                    plt.show(block=False)
                    plt.pause(100)
            else:
                raw_hm = raw_hm.cpu().detach().numpy() * 255
                raw_hm = raw_hm.squeeze()
                inout = inout.cpu().detach().numpy()
                inout = 1 / (1 + np.exp(-inout))
                inout = (1 - inout) * 255
                im_resized = Image.fromarray(raw_hm).resize((width, height), Image.BILINEAR)
                norm_map = np.array(im_resized) - inout
                
                norm_map = np.clip(norm_map, 0, 255).astype(np.uint8)
                # 热力图数据由黑白转为jet色图
                heatmap = cv2.applyColorMap(norm_map, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(frame_raw, 0.8, heatmap, 0.4, 0)
                cv2.imshow("frame",overlay)
                # cv2显示视频流必须加这个,不然就是黑块
                cv2.waitKey(1)