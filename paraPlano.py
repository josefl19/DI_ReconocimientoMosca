# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 22:08:05 2021

@author: josef
"""

#Librerias a usar
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

img = cv2.imread('blanco_1.jpg')
#def caracteristicas(img):
# Imagen a escala de grises
I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Obtencion de la imagen binaria (0 negro y 1 blanco)
umbralA,_ = cv2.threshold(I, 0, 255, cv2.THRESH_OTSU) # umbral automatico por funcion
mascara = np.uint8((I>umbralA)*255) #Mascara con umblal generado por funcion

"""
Etiquetado de objetos y selección del objeto de interés.
Recibe la imagen binaria y devuelve una imagen que estara etiquetada con
los objetos colocados como parametros.
"""
n_mascara = mascara
output = cv2.connectedComponentsWithStats(n_mascara, 4, cv2.CV_32S)
cantObj = output[0]
labels = output[1]
stats = output[2]
n_mascara = (np.argmax(stats[:,4][1:])+1==labels)
n_mascara = ndimage.binary_fill_holes(n_mascara).astype(int)

#Calcular las componentes de cada color de la imagen
rojo = np.sum(n_mascara*img[:,:,0]/255)/np.sum(n_mascara)
verde = np.sum(n_mascara*img[:,:,1]/255)/np.sum(n_mascara)
azul = np.sum(n_mascara*img[:,:,2]/255)/np.sum(n_mascara)

mascara1=np.uint8(n_mascara*255)

contours,hierarchy=cv2.findContours(mascara1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
rect = cv2.minAreaRect(cnt)
box = np.int0(cv2.boxPoints(rect))
box=np.int0(box)
m,n=mascara1.shape
ar1 = np.zeros((m,n))
mascaraRect = cv2.fillConvexPoly(ar1, box, 1)
mascaraRect = np.uint8(mascaraRect.copy()*255)
contoursR,hierarchyR = cv2.findContours(mascaraRect,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cntR = contoursR[0]

centro,dimensiones,rotacion=cv2.minAreaRect(cntR)
tasaAspecto=float(dimensiones[1])/float(dimensiones[0]) if dimensiones[1]<dimensiones[0] else float(dimensiones[0])/float(dimensiones[1])

    #return rojo,verde,azul,tasaAspecto

"""
#Lectura de la imagen
img = cv2.imread('mosca_9.jpg')
datos=[]
etiquetas=[]
for i in range(1,25):
    datos.append(caracteristicas(cv2.imread("mosca_"+str(i)+".jpg")))
    etiquetas.append(1)
    #datos.append(caracteristicas(cv2.imread("manzana"+str(i)+".jpg")))
    #etiquetas.append(-1)
    
datos = np.array(datos)
etiquetas = np.array(etiquetas)

#Graficar
fig =plt.figure()
ax=fig.add_subplot(111,projection='3d')

for i in range(0,72):
    if etiquetas[i]==1:
        ax.scatter(datos[i,0],datos[i,1],datos[i,2],marker='*',c='y')
    #else:
     #   ax.scatter(datos[i,0],datos[i,1],datos[i,2],marker='^',c='r')

#ax.set_xlabel('Rojo')
#ax.set_ylabel('Verde')
#ax.set_zlabel('tasa aspecto')
"""