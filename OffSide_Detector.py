import numpy as np
import argparse
import time
import cv2
import mediapipe as mp

mpDraw=mp.solutions.drawing_utils
mpPose=mp.solutions.pose
pose=mpPose.Pose()

with open('yolov3.txt', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

def distance_to_camera(knownWidth, focalLength, perWidth):
    return (knownWidth * focalLength) / perWidth

def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i- 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label+'('+str(int(confidence*100))+'%)', (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

Xcoord_list=[]
Ycoord_list=[]

X_Lists=[]
Y_Lists=[]

frame=cv2.imread('Sample.jpg')

def Detection():
        global frame
        cv2.imshow('view',frame)
        cv2.waitKey(0)
        global count_detections,detections_list,landmarks_list,Xcoord_list,Ycoord_list
        scale = 0.00392
        (height, width) = frame.shape[:2]
        print(height, width)
        blob = cv2.dnn.blobFromImage(frame, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4
        im2 = frame.copy()
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
                    
                        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            landmarks_list=[]
            Xcoord_list=[]
            Ycoord_list=[]
            
            i = i
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(frame, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

            croppedBox = frame[int(y):int(y + h), int(x):int(x + w)]
            imgRGB=cv2.cvtColor(croppedBox,cv2.COLOR_BGR2RGB)
            results=pose.process(imgRGB)
            
            if results.pose_landmarks:
                mpDraw.draw_landmarks(croppedBox,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
               
                for i in range(len(results.pose_landmarks.landmark)):
                    Xcoord_list.append(results.pose_landmarks.landmark[i].x*w+x)

                X_Lists.append(Xcoord_list)
            
        cv2.imshow("Frame", frame)
        cv2.imwrite('Offside_Detector.jpg',frame)
        cv2.waitKey(0)

Detection()

color = (255, 0, 0)
thickness = 1
cv2.line(frame, (314,0), (314,354), color, thickness)

print('X_Lists',X_Lists)
print(min(X_Lists[0]))
color = (0, 0, 0)
thickness = 1
cv2.line(frame, (int(min(X_Lists[0])),0), (int(min(X_Lists[0])),354), color, thickness)

if min(X_Lists[0])<314:
    print('OFFSIDE')

cv2.imshow("Frame", frame)
cv2.imwrite('Offside_Detector_Final_Output.jpg',frame)
cv2.waitKey(0)

