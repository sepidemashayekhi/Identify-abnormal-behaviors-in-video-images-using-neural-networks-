import os
from unittest import result
from cv2 import KeyPoint, cvtColor
from matplotlib import image
import numpy as np
import cv2

def imread_show_image():
    trainPath='G:/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/'
    dirname=os.listdir(trainPath)
    dirname=dirname[2:]
    images=[]
    for i in range(len(dirname)):
        path=os.path.join(trainPath,dirname[i])
        nameImage=os.listdir(path)
        for name in nameImage:
            pathImage=path+'/'+name
            frame=cv2.imread(pathImage)
            images.append(frame)
        print("load train",dirname[i])
    images=np.array(images)
    return images


def Optical_flow(frame1,frame2):
    frame1_gray=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    frame2_gray=cvtColor(frame2,cv2.COLOR_BGR2GRAY)
    flow=cv2.calcOpticalFlowFarneback(frame1_gray,frame2_gray,None,0.5,3,15,3,5,1.1,0)
    return flow

def Optical_map(data_image):
    first=False
    count=0
    maskHSV=np.zeros_like(data_image[0])
    maskHSV[...,1]=255
    Optical_flow_map=[]
    while count<6800:
        if count==0:
            frame_prv=data_image[0]
            f_frame=frame_prv.copy()
            frame_next=data_image[count+1]
        elif count==6800:
            frame_prv=data_image[6799]
            frame_next=data_image[0]
        elif count>0 and count<6799:
            frame_prv=data_image[count]
            frame_next=data_image[count+1]
        
        flow=Optical_flow(frame_prv,frame_next)
        mag,ang=cv2.cartToPolar(flow[...,0],flow[...,1])
        maskHSV[...,0]=ang*(180/np.pi/2)
        maskHSV[...,2]=cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        maskBGR=cv2.cvtColor(maskHSV,cv2.COLOR_HSV2BGR)
        graymask=cv2.cvtColor(maskBGR,cv2.COLOR_BGR2GRAY)
        _,binary=cv2.threshold(graymask,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


        cv2.imshow("map",maskBGR)
        cv2.imshow("binary",binary)

        Optical_flow_map.append(binary)
        count+=1
        first=True
        if count%200==0:
            print("Finish processing  this level ",count)
        if cv2.waitKey(1)==27:
            break 
    Optical_flow_map=np.array(Optical_flow_map)      
    return Optical_flow_map


def Add(imageData,Map):
    Result=[]
    for i in range(len(Map)):
        result=cv2.bitwise_and(imageData[i],imageData[i],mask=Map[i])
        Result.append(result)
        if i%200==0:
            print("level ",i)
    
    Result=np.array(Result)
    return Result

        
    

