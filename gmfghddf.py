import os,cv2
import numpy as np
import dlib
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import classification_report
from sklearn.externals import joblib   

def getImagesAndLabels(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=np.empty((0,4096))
    Ids=[]
    for imagePath in imagePaths:
        img=cv2.imread(imagePath)
        img2=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img1=cv2.resize(img2,(64,64))
        img1=img1.flatten()
        img1=img1.reshape((1,4096))
        Id=Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faceSamples=np.append(faceSamples,img1,axis=0)
        Ids.append(Id)
    return (faceSamples,Ids)

faces,Ids = getImagesAndLabels('/Users/amitrai/Documents/attendance/photossssss')
print(faces.shape)
knn = KNN(n_neighbors = 3) 
knn.fit(faces,Ids)
joblib.dump(knn, '/Users/amitrai/Documents/attendance/filename.pkl')
print('Successfully Trained')
knn_from_joblib = joblib.load('filename.pkl')  
  # Use the loaded model to make predictions 
facesTest,IdsTest=getImagesAndLabels('/Users/amitrai/Documents/attendance/photosssssTest')

predictions=knn_from_joblib.predict(facesTest)
print(classification_report(IdsTest,predictions))




