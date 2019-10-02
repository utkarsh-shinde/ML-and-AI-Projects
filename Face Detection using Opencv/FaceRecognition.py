import cv2 as cv2
import os as os
import numpy as np 

#Returns rectangle for face detected alongwith gray scale image
def faceRecognition(test_image):
	gray_image = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
	haar_cascade = cv2.CascadeClassifier('./HaarCascade/haarcascade_frontalface_default.xml')
	faces = haar_cascade.detectMultiScale(gray_image,scaleFactor=1.32,minNeighbors=5)

	return faces,gray_image

#Returns part of gray_img which is face alongwith its label/ID
def labels_for_training_data(directory):
	faces=[]
	faceID=[]
	for path,subDirNames,fileNames in os.walk(directory):
		for fileName in fileNames:
			if fileName.startswith("."):
				print("skipping system files")#Skipping files that startwith .
				continue
			id = os.path.basename(path)
			img_path = os.path.join(path,fileName)
			print("Image Path : ",img_path)
			print("ID : ",id)
			test_image = cv2.imread(img_path)
			if test_image is None:
				print("Image not loaded ")
				continue
			faces_rect,gray_img = faceRecognition(test_image)#Calling faceDetection function to return faces detected in particular image
			if len(faces_rect)!=1:
				continue
			(x,y,w,h) = faces_rect[0]
			roi_gray = gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from grayscale image
			faces.append(roi_gray)
			faceID.append(int(id))
	return faces,faceID


#Trains haar classifier and takes faces,faceID returned by previous function as its arguments
def train_classifier(faces,faceID):
	face_recognizer = cv2.face.LBPHFaceRecognizer_create()
	face_recognizer.train(faces,np.array(faceID))
	return face_recognizer

#Draws bounding boxes around detected face in image
def draw_rect(test_img,face):
	(x,y,w,h) = face
	cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=5)

#Writes name of person for detected label
def display_text(test_img,text,x,y):
	cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,2,(0,255,255),2)





