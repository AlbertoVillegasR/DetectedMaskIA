from asyncio.tasks import sleep
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

# BUILDING MODEL TO CLASSIFY BETWEEN MASK AND NO MASK

# model=Sequential()
# model.add(Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
# model.add(MaxPooling2D() )
# model.add(Conv2D(32,(3,3),activation='relu'))
# model.add(MaxPooling2D() )
# model.add(Conv2D(32,(3,3),activation='relu'))
# model.add(MaxPooling2D() )
# model.add(Flatten())
# model.add(Dense(100,activation='relu'))
# model.add(Dense(1,activation='sigmoid'))

# model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# from keras.preprocessing.image import ImageDataGenerator
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)

# test_datagen = ImageDataGenerator(rescale=1./255)

# training_set = train_datagen.flow_from_directory(
#         'train',
#         target_size=(150,150),
#         batch_size=16 ,
#         class_mode='binary')

# test_set = test_datagen.flow_from_directory(
#         'test',
#         target_size=(150,150),
#         batch_size=16,
#         class_mode='binary')

# model_saved=model.fit_generator(
#         training_set,
#         epochs=10,
#         validation_data=test_set,

#         )

# model.save('mymodel.h5',model_saved)

#To test for individual images

# test_image=image.load_img('C:/Users/Karan/Desktop/ML Datasets/Face Mask Detection/Dataset/test/without_mask/30.jpg',target_size=(150,150,3))
# test_image=image.load_img(r'C:\Users\Karan\Pictures\Camera Roll/21.jpg',
#                           target_size=(150,150,3))
# test_image
# test_image=image.img_to_array(test_image)
# test_image=np.expand_dims(test_image,axis=0)
# type(mymodel.predict_classes(test_image)[0][0])

# IMPLEMENTING LIVE DETECTION OF FACE MASK



def convert_date (date_import):
    date = date_import.split()
    date = date[0]
    date = date.split('-')
    date = date[2]+"/"+date[1]+"/"+ date[0]
    return date

def upload_firebase(interaction, date, mask):
    result = db.collection(u'interaction').add({
            u'Interration': bool(interaction),
            u'Create_at': str(date),
    })
    if(result):
        print("interaction up")

    result = db.collection(u'clients').add({
            u'Mask': bool(mask),
            u'Create_at': str(date),
    })
    if(result):
        print("person up")

def alarm ():
    winsound.Beep(4400,1000)
    #os.system("alarm.mp3")

def main():
    mymodel=load_model('mymodel.h5')
    cap=cv2.VideoCapture(0)
    face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    var=0
    no_mask = None
    now=time.time()
    while cap.isOpened():
        check = None
        _,img=cap.read()
        face=face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)
        active = True
        for(x,y,w,h) in face:
            datet=str(datetime.datetime.now())
            face_img = img[y:y+h, x:x+w]
            cv2.imwrite('temp.jpg',face_img)
            test_image=image.load_img('temp.jpg',target_size=(150,150,3))
            test_image=image.img_to_array(test_image)
            test_image=np.expand_dims(test_image,axis=0)
            pred=mymodel.predict(test_image)[0][0]
            if pred==1:
                var += 1
                image_name = 'Images/Without/'+str(var)+'.jpg'
                cv2.imwrite(image_name,face_img)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
                cv2.putText(img,'SIN_MASCARILLA',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                no_mask = True
                threading.Thread(target=alarm).start()
                
            else:
                cv2.putText(img,'Adelante... Gracias!',(30,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
                var += 1
                image_name = 'Images/Mask/'+str(var)+'.jpg'
                cv2.imwrite(image_name,face_img)
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
                cv2.putText(img,'CON_MASCARILLA',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)

            if(no_mask == True):
                files_mask = glob.glob('Images/Mask/*.jpg')
                if not files_mask:
                    cv2.putText(img,'USE CORRECTAMENTE LA MASCARILLA',(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
                    
                else:
                    date = convert_date(datet)
                    threading.Thread(target= upload_firebase, args=(True,date,no_mask)).start()
                    no_mask = False
                    files_mask = glob.glob('Images/Mask/*.jpg')
                    files_no_mask = glob.glob('Images/Without/*.jpg')
                    for i in files_mask: 
                        os.remove(i)
                    for i in files_no_mask: 
                        os.remove(i)

        cv2.imshow('img',img)
        if cv2.waitKey(1)==ord('q'):
            files_mask = glob.glob('Images/Mask/*.jpg')
            files_no_mask = glob.glob('Images/Without/*.jpg')
            for i in files_mask: 
                os.remove(i)
            for i in files_no_mask: 
                os.remove(i)
            break    
    cap.release()
    cv2.destroyAllWindows()

main()