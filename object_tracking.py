from jetbot import ObjectDetector
from jetbot import Camera
import cv2
import numpy as np


model = ObjectDetector('ssd_mobilenet_v2_coco.engine')
camera = Camera.instance(width=300, height=300)
detections = model(camera.value)

while True:
    
    

        
    # compute all detected objects
    detections = model(camera.value)
    img = camera.value
    mask = np.zeros(img.shape[:2],dtype = np.uint8)
    # draw all detections on image
    for det in detections[0]:
        bbox = det['bbox']         
        cv2.rectangle(mask, (int(300 * bbox[0]), int(300 * bbox[1])), (int(300 * bbox[2]), int(300 * bbox[3])), (255, 0, 0), 2)
      
    orb = cv2.ORB_create()
    kp = orb.detect(camera.value,mask)
    kp,des = orb.compute(camera.value,kp)
    img = cv2.drawKeypoints(camera.value,kp,camera.value)
    cv2.imshow('detects',img)
    
    cv2.waitKey(1)
    

