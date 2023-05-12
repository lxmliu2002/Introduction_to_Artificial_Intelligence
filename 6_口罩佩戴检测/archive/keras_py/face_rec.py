from tensorflow.keras.applications.imagenet_utils import preprocess_input

import numpy as np
import cv2 as cv
import os

from keras_py.mobileNet import MobileNet
from keras_py.mtcnn import mtcnn
import keras_py.utils as utils



class face_rec():
    def __init__(self,pnet_path,rnet_path,onet_path):
        #创建mtcnn对象来检测人脸
        self.mtcnn_model = mtcnn(pnet_path,rnet_path,onet_path)
        self.threshold = [0.5,0.6,0.8]
        self.Crop_HEIGHT = 160
        self.Crop_WIDTH = 160

    def recognize(self,draw):
        height,width,_ = np.shape(draw)
        draw_rgb = cv.cvtColor(draw,cv.COLOR_BGR2RGB)
        rectangles = self.mtcnn_model.detectFace(draw_rgb,self.threshold)
        if len(rectangles)==0:
            return

        rectangles = np.array(rectangles,dtype=np.int32)
        rectangles[:,0] = np.clip(rectangles[:,0],0,width)
        rectangles[:,1] = np.clip(rectangles[:,1],0,height)
        rectangles[:,2] = np.clip(rectangles[:,2],0,width)
        rectangles[:,3] = np.clip(rectangles[:,3],0,height)

        rectangles_temp = utils.rect2square(np.array(rectangles,dtype=np.int32))
        rectangles_temp[:,0] = np.clip(rectangles_temp[:,0],0,width)
        rectangles_temp[:,1] = np.clip(rectangles_temp[:,1],0,height)
        rectangles_temp[:,2] = np.clip(rectangles_temp[:,2],0,width)
        rectangles_temp[:,3] = np.clip(rectangles_temp[:,3],0,height)

        for rectangle in rectangles_temp:
            # 获取landmark在小图中的坐标
            landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160
            # 截取图像
            crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img = cv.resize(crop_img,(self.Crop_HEIGHT,self.Crop_WIDTH))
            # 对齐
            new_img,_ = utils.Alignment_1(crop_img,landmark)
            # 归一化
            new_img = preprocess_input(np.reshape(np.array(new_img,np.float64),[1,self.Crop_HEIGHT,self.Crop_WIDTH,3]))

        rectangles = rectangles[:,0:4]

        #draw = cv2.cvtColor(draw, cv2.COLOR_RGB2BGR)
        for (left, top, right, bottom) in rectangles:
            cv.rectangle(draw, (left, top), (right, bottom), (255, 0, 0), 2)
        return draw


class mask_rec():
    def __init__(self,model_path=None):
        # 预训练模型路径
        pnet_path = "./datasets/5f680a696ec9b83bb0037081-momodel/data/keras_model_data/pnet.h5"
        rnet_path = "./datasets/5f680a696ec9b83bb0037081-momodel/data/keras_model_data/rnet.h5"
        onet_path = "./datasets/5f680a696ec9b83bb0037081-momodel/data/keras_model_data/onet.h5"
        classes_path = "./datasets/5f680a696ec9b83bb0037081-momodel/data/keras_model_data/classes.txt"
        # 创建 mtcnn 对象 检测图片中的人脸
        self.mtcnn_model = mtcnn(pnet_path,rnet_path,onet_path)
        # 门限函数
        self.threshold = [0.5,0.6,0.8]

        self.Crop_HEIGHT = 160
        self.Crop_WIDTH = 160
        # self.classes_path = "./data/model_data/classes.txt"
        self.classes_path = classes_path
        self.NUM_CLASSES = 2
        self.mask_model = MobileNet(input_shape=[self.Crop_HEIGHT,self.Crop_WIDTH,3],classes=self.NUM_CLASSES)
        self.mask_model.load_weights(model_path)
        # self.mask_model.load_weights("./results/temp.h5")
        self.class_names = self._get_class()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def recognize(self,draw):
        #-----------------------------------------------#
        #   人脸识别
        #   先定位，再进行匹配
        #-----------------------------------------------#
        height,width,_ = np.shape(draw)
        draw = cv.cvtColor(np.asarray(draw),cv.COLOR_RGB2BGR)
        draw_rgb = cv.cvtColor(draw,cv.COLOR_BGR2RGB)

        # 检测人脸
        rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)
        if len(rectangles)==0:
            return draw,0,0

        rectangles = np.array(rectangles,dtype=np.int32)
        rectangles[:,0] = np.clip(rectangles[:,0],0,width)
        rectangles[:,1] = np.clip(rectangles[:,1],0,height)
        rectangles[:,2] = np.clip(rectangles[:,2],0,width)
        rectangles[:,3] = np.clip(rectangles[:,3],0,height)

        rectangles_temp = utils.rect2square(np.array(rectangles,dtype=np.int32))
        rectangles_temp[:,0] = np.clip(rectangles_temp[:,0],0,width)
        rectangles_temp[:,1] = np.clip(rectangles_temp[:,1],0,height)
        rectangles_temp[:,2] = np.clip(rectangles_temp[:,2],0,width)
        rectangles_temp[:,3] = np.clip(rectangles_temp[:,3],0,height)
        # 转化成正方形
        #-----------------------------------------------#
        #   对检测到的人脸进行编码
        #-----------------------------------------------#
        classes_all = []
        for rectangle in rectangles_temp:

            # 获取landmark在小图中的坐标
            landmark = (np.reshape(rectangle[5:15],(5,2)) - np.array([int(rectangle[0]),int(rectangle[1])]))/(rectangle[3]-rectangle[1])*160
            # 截取图像
            crop_img = draw_rgb[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            crop_img = cv.resize(crop_img,(self.Crop_HEIGHT,self.Crop_WIDTH))
            # 对齐
            new_img,_ = utils.Alignment_1(crop_img,landmark)
            # 归一化
            new_img = preprocess_input(np.reshape(np.array(new_img,np.float64),[1,self.Crop_HEIGHT,self.Crop_WIDTH,3]))

            classes = self.class_names[np.argmax(self.mask_model.predict(new_img)[0])]
            classes_all.append(classes)

        rectangles = rectangles[:,0:4]
        all_num = rectangles.shape[0]
        mask_num = 0
        #-----------------------------------------------#
        #   画框
        #-----------------------------------------------#
        for (left, top, right, bottom), c in zip(rectangles,classes_all):
            if c == "YES":
                mask_num = mask_num+1
                cv.rectangle(draw, (left, top), (right, bottom), (0, 0, 255), 2)
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(draw, c, (left , bottom - 15), font, 0.75, (255, 255, 255), 2)
            else:
                cv.rectangle(draw, (left, top), (right, bottom), (255, 0, 0), 2)
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(draw, c, (left , bottom - 15), font, 0.75, (255, 255, 255), 2)
        return draw,all_num,mask_num