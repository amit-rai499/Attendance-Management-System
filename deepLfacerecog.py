# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2
import os
# grab the paths to the input images in our dataset
print("[INFO] quantifying faces...")
imagePaths=[os.path.join('/Users/amitrai/Documents/attendance/photossssss',f) for f in os.listdir('/Users/amitrai/Documents/attendance/photossssss')]
print(len(imagePaths)) 
# initialize the list of known encodings and known names
knownEncodings = []
knownIds = []
# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person Id from the image path
	print(imagePath)
	print("[INFO] processing image {}/{}".format(i + 1,len(imagePaths)))
	Id=Id=int(os.path.split(imagePath)[-1].split(".")[1])
 
	# load the input image and convert it from BGR (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb,model="hog")
 
	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)
 
	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and
		# encodings
		knownEncodings.append(encoding)
		knownIds.append(Id)
# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "Ids": knownIds}
f = open('enkode.pickle', "wb")
f.write(pickle.dumps(data))
f.close()