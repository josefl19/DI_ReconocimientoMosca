# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 21:19:12 2021

@author: josef
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten

import cv2
import numpy as np

modelo=Sequential()
modelo.add(Convolution2D(32, (3,3),input_shape=(480,480,3),activation='relu'))
modelo.add(MaxPooling2D(pool_size=((2,2))))
modelo.add(Flatten())
modelo.add(Dense(64,activation='relu'))
modelo.add(Dense(25,activation='relu'))
modelo.add(Dense(1,activation='sigmoid'))
modelo.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


x_train=[]
y_train=[]
x_test=[]
y_test=[]

dataTr=[]


import glob
import os
for filename in glob.glob(os.path.join('datos/entrenamiento/mosca','*.jpg')):
    dataTr.append([1,cv2.imread(filename)])
for filename in glob.glob(os.path.join('datos/entrenamiento/otros','*.jpg')):
    dataTr.append([0,cv2.imread(filename)])    
    
    
from random import shuffle

shuffle(dataTr)

for i,j in dataTr:
    y_train.append(i)
    x_train.append(j)    
x_train=np.array(x_train)
y_train=np.array(y_train)    

#comprobar=[]
for filename in glob.glob(os.path.join('datos/test/mosca','*.jpg')):
    x_test.append(cv2.imread(filename))
    #comprobar.append([1,cv2.imread(filename),filename])
    y_test.append(1)
    
for filename in glob.glob(os.path.join('datos/test/otros','*.jpg')):
    x_test.append(cv2.imread(filename))
    #comprobar.append([2,cv2.imread(filename),filename])
    y_test.append(0)

x_test=np.array(x_test)
y_test=np.array(y_test)

modelo.fit(x_train,y_train,batch_size=32,epochs=20,validation_data=(x_test, y_test))


"""
ruta='datos/test/mosca4.jpg'
I=cv2.imread(ruta)

if round(modelo.predict(np.array([I]))[0][0])==1:
    print("Su cultivo tiene mosca blanca")
    cv2.imshow('Mosca blanca',I)
else:
    print("No es mosca blanca, se trata de otro insecto")
    cv2.imshow('Negativo a mosca blanca',I)
"""