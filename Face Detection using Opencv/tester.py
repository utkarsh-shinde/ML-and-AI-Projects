import cv2 as cv2
import os as os
import numpy as np 
import FaceRecognition as fr 

test_img = cv2.imread("./TestImages/1462332-KanganaRanautcopy-1500543743.jpg") # test image path

faces_detected,gray_img = fr.faceRecognition(test_img)

print("Faces Detected : ",faces_detected)


if os.path.exists("trainingData.yml"):
	face_recognizer = cv2.face.LBPHFaceRecognizer_create()
	face_recognizer.read("trainingData.yml")

	print("No need to load training data as it exists")
else:
	faces,faceID = fr.labels_for_training_data("trainingImages")
	face_recognizer = fr.train_classifier(faces,faceID)
	face_recognizer.save("trainingData.yml")



#creating dictionary containing names for each label
name={0:"Priyanka",1:"Kangana",2:"Dhoni"}

for face in faces_detected:
	(x,y,w,h) = face
	roi_gray = gray_img[y:y+w,x:x+h]
	label,confidence = face_recognizer.predict(roi_gray)
	print("confidence : ",confidence)
	print("label : ",label)
	fr.draw_rect(test_img,face)
	predicted_name = name[label]
	#If confidence more than 40 then don't print predicted face text on screen
	if(confidence>40):
		continue
    
	fr.display_text(test_img,predicted_name,x,y)

resized_img = cv2.resize(test_img,(600,600))
cv2.imshow("Face Detection Window",resized_img)
cv2.waitKey(0)#Waits indefinitely until a key is pressed
cv2.destroyAllWindows