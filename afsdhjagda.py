# Import OpenCV2 for image processing
import cv2
import os
import dlib
import time
    
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
face_id=input('enter your id: ')

# Start capturing video 
vid_cam = cv2.VideoCapture(0)

# Detect object in video stream using Dlib's Frontal Face
face_detector = dlib.get_frontal_face_detector()

# Initialize sample face image
count = 0

assure_path_exists("/Users/amitrai/Documents/attendance/photossssss")

# Start looping
while(True):

    # Capture video frame
    _, image_frame = vid_cam.read()
    
    # Convert frame to grayscale
    rgb_image = cv2.cvtColor(image_frame, cv2.COLOR_RGB2GRAY)

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
        
        # Increment sample face image
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("/Users/amitrai/Documents/attendance/photossssss/User." + str(face_id) + '.' + str(count) + ".jpg", rgb_image[y:y+h,x:x+w])

        # Display the video frame, with bounded rectangle on the person's face
        cv2.imshow('frame', image_frame)

    # To stop taking video, press 'q' for at least 100ms
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

    # If image taken reach 30, stop taking video
    elif count>=30:
        print("Successfully Captured")
        break


