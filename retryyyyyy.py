# Import OpenCV2 for image processing
import cv2
import os
import dlib
import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
    

# Start capturing video 
vid_cam = cv2.VideoCapture(0)

# Detect object in video stream using Dlib's Frontal Face
face_detector = dlib.get_frontal_face_detector()

# Start looping
while(True):

    # Capture video frame
    _, image_frame = vid_cam.read()
    
    # Convert frame to grayscale
    rgb_image = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Detect frames of different sizes, list of faces rectangles
    dets = face_detector(rgb_image)

    # Loops for each faces
    for det in dets:
        # take a bounding predicted by dlib and convert it
	    # to the format (x, y, w, h) as we would normally do
	    # with OpenCV
        #Source for conversion :https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
        x = det.left()
        y = det.top()
        w = det.right() - x
        h = det.bottom() - y

        # Crop the image frame into rectangle
        cv2.rectangle(image_frame,(x,y), (x+w,y+h), (255,0,0), 2)
        

        nparray=rgb_image[y:y+h,x:x+w]
        print(nparray.shape)
        img = Image.fromarray( nparray , 'L')
        print(img.size)
        print(img)
        #img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img1=cv2.resize(img,(64,64))
        print(img1.size)
        print(img1)
        img1=img1.flatten()
        img1=img1.reshape((1,4096))
        print(img1)

        # Display the video frame, with bounded rectangle on the person's face
        cv2.imshow('frame', image_frame)

    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

# Stop video
vid_cam.release()

# Close all started windows
cv2.destroyAllWindows()
