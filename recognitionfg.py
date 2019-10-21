import cv2
import numpy as np
import dlib
from sklearn.externals import joblib

vid_cam=cv2.VideoCapture(0)

face_detector=dlib.get_frontal_face_detector()
while(True):

    # Capture video frame
    _, image_frame = vid_cam.read()
    # Convert frame to grayscale
    rgb_image = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    # Detect frames of different sizes, list of faces rectangles
    dets=face_detector(rgb_image) 
    print(dets)
    # Loops for each faces
    for det in dets:
        # take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
        #Source for conversion :https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
        print(dets[0])
        print(dets[1])
        x = det.left()
        y = det.top()
        w = det.right() - x
        h = det.bottom() - y

        # Crop the image frame into rectangle
        cv2.rectangle(image_frame,(x,y), (x+w,y+h), (255,0,0), 2)
        img=rgb_image[y:y+h,x:x+w]
        # Display the video frame, with bounded rectangle on the person's face
        cv2.imshow('frame', img)
        img1=cv2.resize(img,(64,64))
        
        img1=img1.flatten()
        img1=img1.reshape((1,4096))
        #faceSamples=np.append(faceSamples,img1,axis=0)
        knn_from_joblib = joblib.load('filename.pkl')  
  # Use the loaded model to make predictions 
        predictions=knn_from_joblib.predict(img1)
        print(predictions)
        #if(predictions[0]==1):
        #    print('Amit')
        #elif(predictions[0]==2):
        #    print('sumit')
        #else:
 #
 #      # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
