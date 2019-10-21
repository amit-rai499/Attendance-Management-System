# import the necessary packages
from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import pickle
import time
import cv2

# load the known faces and embeddings
print("[INFO] loading encodings...")
data = pickle.loads(open('/Users/amitrai/Documents/attendance/enkode.pickle', "rb").read())
 
# initialize the video stream and pointer to output video file, then
# allow the camera sensor to warm up

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
writer = None
#time.sleep(2.0)
# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()
	
    # convert the input frame from BGR to RGB then resize it to have
    # a width of 750px (to speedup processing)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = imutils.resize(frame, width=750)
    r = frame.shape[1] / float(rgb.shape[1])
 
    # detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input frame, then compute
	# the facial embeddings for each face
    boxes = face_recognition.face_locations(rgb,model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    Id = []
    	# loop over the facial embeddings
    for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
        matches = face_recognition.compare_faces(data["encodings"],encoding)
        name = "Unknown"
 
		# check to see if we have found a match
        if True in matches:
			# find the indexes of all matched faces then initialize a
			# dictionary to count the total number of times each face
			# was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
 
			# loop over the matched indexes and maintain a count for
			# each recognized face face
            for i in matchedIdxs:
                name = data["Ids"][i]
                print(name)
                counts[name] = counts.get(name, 0) + 1
 
			# determine the recognized face with the largest number
			# of votes (note: in the event of an unlikely tie Python
			# will select first entry in the dictionary)
            name = max(counts, key=counts.get)
		
		# update the list of names
        Id.append(name)
        # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, Id):
		# rescale the face coordinates
        top = int(top * 4)
        right = int(right * 4)
        bottom = int(bottom * 4)
        left = int(left * 4)
        print(name)
        
 
	# draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom),(0, 255, 0), 2)
        #y = top - 15 if top - 15 > 15 else top + 15
        #cv2.putText(frame, name, (left, y),cv2.FONT_HERSHEY_PLAIN,0.75, (0, 255, 0), 2)
	#display the output to the screen
    key = cv2.waitKey(1) & 0xFF
 
		# if the `q` key was pressed, break from the loop
    if key == ord("q"):
	    break