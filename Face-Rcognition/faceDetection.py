import cv2,os
import numpy as np
import imutils
import urllib.request

url = 'http://192.168.1.7:8080/shot.jpg'
haar_file="haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)
datasets = "dataset"
print('Training....')
(images , labels, names, id) = ([],[],{},0)

for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets,subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' +filename
            label = id
            images.append(cv2.imread(path,0))
            labels.append(int(label))
        id +=1

(images, labels) = [np.array(lis) for lis in [images, labels]]
print(images,labels)
(width,height) =(130,100)

#classifier
model = cv2.face.LBPHFaceRecognizer_create()
#model = cv2.face.FisherFaceRecognizer_create()
model.train(images, labels)


cnt=0

while True:
    imgPath = urllib.request.urlopen(url)
    imgNp = np.array(bytearray(imgPath.read()), dtype=np.uint8)
    img = cv2.imdecode(imgNp, -1)
    img = imutils.resize(img, width=450)
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayImg,1.3,5)
    for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
            face = grayImg[y:y+h,x:x+w]
            face_resize = cv2.resize(face,(width,height))

            prediction = model.predict(face_resize)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            if prediction[1] <800:
                cv2.putText(img,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255))
                cnt =0
            else:
                cnt +=1
                cv2.putText(img,'Unknown' % (names[prediction[0]],prediction[1]),(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255))
                if(cnt >100):
                    print("Unknown Person")
                    cv2.imwrite('unknown.jpg',img)
                    cnt =0            
    cv2.imshow("FaceRecognization",img)
    key = cv2.waitKey(10)
    if key == 27:
        break
webcam.release()
cv2.destroyAllWindows()
