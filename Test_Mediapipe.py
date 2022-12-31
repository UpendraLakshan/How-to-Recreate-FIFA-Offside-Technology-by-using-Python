import cv2
import mediapipe as mp

mpDraw=mp.solutions.drawing_utils
mpPose=mp.solutions.pose
pose=mpPose.Pose()

img=cv2.imread('Test_Mediapipe.jpg')
cv2.imshow('img',img)

imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

results=pose.process(imgRGB)
print(results.pose_landmarks)

if results.pose_landmarks:
	mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)


cv2.imshow('out',img)
cv2.imwrite('MP_Out.jpg',img)
cv2.waitKey(0)








