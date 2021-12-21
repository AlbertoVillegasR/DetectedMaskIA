from typing import Any
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import cv2
import threading
import os
import glob
import winsound
import asyncio


##FIREBASE##
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import time, datetime

# Credentials for firebase
cred = credentials.ApplicationDefault()
cred = credentials.Certificate("key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def convert_date (date_import):
    date = date_import.split()
    date = date[0]
    date = date.split('-')
    date = date[2]+"/"+date[1]+"/"+ date[0]
    return date

def remove_files_wmask():
    files_no_mask = glob.glob('Images/Without/*.jpg')
    for i in files_no_mask: 
        os.remove(i)

def remove_files_mask():
    files_mask = glob.glob('Images/Mask/*.jpg')
    for i in files_mask: 
        os.remove(i)


def upload_firebase_interaction(interaction, date):
    result = db.collection(u'interaction').add({
            u'Interration': bool(interaction),
            u'Create_at': str(date),
    })
    if(result):
        print("interaction up")

def upload_firebase_person(date, mask):
    result = db.collection(u'clients').add({
            u'Mask': bool(mask),
            u'Create_at': str(date),
    })
    if(result):
        print("person up")

def alarm ():
    winsound.Beep(4400,1000)

def main():
    mymodel=load_model('mymodel.h5')
    cap=cv2.VideoCapture(0)
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    var=0
    no_mask = None
    detect=[]
    mask= None
    while cap.isOpened():
        _,img=cap.read()
        face=face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)
        for (x,y,w,h) in face:
            datet=str(datetime.datetime.now())
            date = convert_date(datet)
            face_img = img[y:y+h, x:x+w]
            cv2.imwrite('temp.jpg',face_img)
            test_image=image.load_img('temp.jpg',target_size=(150,150,3))
            test_image=image.img_to_array(test_image)
            test_image=np.expand_dims(test_image,axis=0)
            pred=mymodel.predict(test_image)[0][0]
            print(cap)
            if pred==1:
                var += 1
                image_name = 'Images/Without/'+str(var)+'.jpg'
                cv2.imwrite(image_name,face_img)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
                cv2.putText(img,'SIN_MASCARILLA',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                no_mask = True
                mask = False
                detect.append(pred)
                detect.append(pred)
                detect.append(pred)
                threading.Thread(target=alarm).start()
                
            else:
                cv2.putText(img,'Adelante... Gracias!',(30,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
                cv2.putText(img,'CON_MASCARILLA',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
                var += 1
                image_name = 'Images/Mask/'+str(var)+'.jpg'
                cv2.imwrite(image_name,face_img)
                mask = True
                detect.append(pred)

            if(no_mask == True):
                files_mask = glob.glob('Images/Mask/*.jpg')
                if not files_mask:
                    cv2.putText(img,'USE CORRECTAMENTE LA MASCARILLA',(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                    if(pred is None):
                        threading.Thread(target= upload_firebase_interaction, args=(False,date)).start()
                        threading.Thread(target= upload_firebase_person, args=(date,False)).start()
                else:
                    threading.Thread(target= upload_firebase_interaction, args=(True,date)).start()
                    # threading.Thread(target= upload_firebase_person, args=(date,no_mask)).start()
                    no_mask = False
                    detect=[]
                    detect.append(1)


            list=len(detect)
            if(list == 2):
                cond = detect[0]
                print(cond)
                if(cond == 1):
                    mask = False
                    threading.Thread(target= upload_firebase_person, args=(date,mask)).start()
                else:
                    threading.Thread(target= upload_firebase_person, args=(date,mask)).start()

            remove_files_mask()
            remove_files_wmask()

        cv2.imshow('img',img)
        list=len(detect)
        if(list >= 5):
            cond = detect.pop()
            if(cond == 0):
                print("Ingresa Mascarilla")
                time.sleep(2)
                detect=[]
        if cv2.waitKey(1)==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

main()