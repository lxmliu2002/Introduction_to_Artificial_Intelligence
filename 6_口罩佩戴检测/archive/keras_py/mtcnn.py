from tensorflow.keras.layers import Conv2D,Input,MaxPool2D,Flatten,Dense,Permute
from tensorflow.keras.layers import PReLU
from tensorflow.keras.models import Model
import numpy as np
import cv2 as cv

import keras_py.utils as utils


def create_Pnet(weight_path):
    # h,w
    input = Input(shape=[None, None, 3])

    # h,w,3 -> h/2,w/2,10
    x = Conv2D(10, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1,2],name='PReLU1')(x)
    x = MaxPool2D(pool_size=2)(x)

    # h/2,w/2,10 -> h/2,w/2,16
    x = Conv2D(16, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='PReLU2')(x)
    # h/2,w/2,32
    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='PReLU3')(x)

    # h/2, w/2, 2
    classifier = Conv2D(2, (1, 1), activation='softmax', name='conv4-1')(x)
    # 无激活函数，线性。
    # h/2, w/2, 4
    bbox_regress = Conv2D(4, (1, 1), name='conv4-2')(x)

    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model

def create_Rnet(weight_path):
    input = Input(shape=[24, 24, 3])
    # 24,24,3 -> 11,11,28
    x = Conv2D(28, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1, 2], name='prelu1')(x)
    x = MaxPool2D(pool_size=3,strides=2, padding='same')(x)

    # 11,11,28 -> 4,4,48
    x = Conv2D(48, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    # 4,4,48 -> 3,3,64
    x = Conv2D(64, (2, 2), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    # 3,3,64 -> 64,3,3
    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)
    # 576 -> 128
    x = Dense(128, name='conv4')(x)
    x = PReLU( name='prelu4')(x)
    # 128 -> 2 128 -> 4
    classifier = Dense(2, activation='softmax', name='conv5-1')(x)
    bbox_regress = Dense(4, name='conv5-2')(x)
    model = Model([input], [classifier, bbox_regress])
    model.load_weights(weight_path, by_name=True)
    return model

def create_Onet(weight_path):
    input = Input(shape = [48,48,3])

    x = Conv2D(32, (3, 3), strides=1, padding='valid', name='conv1')(input)
    x = PReLU(shared_axes=[1,2],name='prelu1')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='prelu2')(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    x = Conv2D(64, (3, 3), strides=1, padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='prelu3')(x)
    x = MaxPool2D(pool_size=2)(x)

    x = Conv2D(128, (2, 2), strides=1, padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1,2],name='prelu4')(x)

    x = Permute((3,2,1))(x)
    x = Flatten()(x)
    x = Dense(256, name='conv5') (x)
    x = PReLU(name='prelu5')(x)

    classifier = Dense(2, activation='softmax',name='conv6-1')(x)
    bbox_regress = Dense(4,name='conv6-2')(x)
    landmark_regress = Dense(10,name='conv6-3')(x)

    model = Model([input], [classifier, bbox_regress, landmark_regress])
    model.load_weights(weight_path, by_name=True)

    return model


class mtcnn():
    def __init__(self,pnet_path,rnet_path,onet_path):
        self.Pnet = create_Pnet(pnet_path)
        self.Rnet = create_Rnet(rnet_path)
        self.Onet = create_Onet(onet_path)

    def detectFace(self,img,threshold):
        #归一化
        copy_img = (img.copy()-127.5)/127.5
        origin_h,origin_w,_ = copy_img.shape
        #计算原始输入图像每一次缩放的比例/实现图像金字塔
        scales = utils.calculateScales(img)
        out = []

        # pnet部分
        for scale in scales:
            hs = int(origin_h*scale)
            ws = int(origin_w*scale)
            scale_img = cv.resize(copy_img,(ws,hs))
            inputs = scale_img.reshape(1,*scale_img.shape)
            output = self.Pnet.predict(inputs)
            out.append(output)

        image_num = len(scales)
        rectangles = []
        for i in range(image_num):
            #有人脸的概率
            cls_prob = out[i][0][0][:,:,1]
            #其对应的框的位置
            roi = out[i][1][0]
            #取出每个缩放后图片的长宽
            out_h,out_w = cls_prob.shape
            out_side = max(out_h,out_w)
            #映射回原图
            rectangle = utils.detect_face_12net(cls_prob, roi, out_side, 1 / scales[i], origin_w, origin_h, threshold[0])
            rectangles.extend(rectangle)

        #非极大值抑制
        rectangles = utils.NMS(rectangles,0.7)

        if len(rectangles)==0:
            return rectangles

        #Rnet部分
        predict_24_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]),int(rectangle[0]):int(rectangle[2])]
            scale_img = cv.resize(crop_img,(24,24))
            predict_24_batch.append(scale_img)

        predict_24_batch = np.array(predict_24_batch)
        out = self.Rnet.predict(predict_24_batch)

        #可信度
        cls_prob = out[0]
        cls_prob = np.array(cls_prob)
        roi_prob = out[1]
        roi_prob = np.array(roi_prob)
        rectangles = utils.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])
        if len(rectangles)==0:
            return rectangles

        #Onet部分
        predict_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
            scale_img = cv.resize(crop_img, (48, 48))
            predict_batch.append(scale_img)

        predict_batch = np.array(predict_batch)
        output = self.Onet.predict(predict_batch)
        cls_prob = output[0]
        roi_prob = output[1]
        pts_prob = output[2]

        rectangles = utils.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])
        return rectangles
