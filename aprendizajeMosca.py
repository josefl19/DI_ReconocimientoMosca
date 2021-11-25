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

#img = cv2.imread('mosca_9.jpg')
def caracteristicas(img):
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
    rojo = 255*(np.sum(n_mascara*img[:,:,0]/255)/np.sum(n_mascara))
    verde = 255*(np.sum(n_mascara*img[:,:,1]/255)/np.sum(n_mascara))
    azul = 255*(np.sum(n_mascara*img[:,:,2]/255)/np.sum(n_mascara))
    
    mascara1=np.uint8(n_mascara*255)
    
    promedioB = (rojo+azul+verde)/3
    promedioA = (rojo+azul)/2

    
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
    
    return tasaAspecto, promedioB, promedioA

#Lectura de la imagen
img = cv2.imread('mosca_9.jpg')
datos=[]
etiquetas=[]
for i in range(1,12):
    datos.append(caracteristicas(cv2.imread("mosca_"+str(i)+".jpg")))
    etiquetas.append(1)
    datos.append(caracteristicas(cv2.imread("cultivo_"+str(i)+".jpg")))
    etiquetas.append(-1)
    
datos = np.array(datos)
etiquetas = np.array(etiquetas)


#Graficar
fig =plt.figure()
ax=fig.add_subplot(111,projection='3d')

for i in range(0,22):
    if etiquetas[i]==1:
        ax.scatter(datos[i,0],datos[i,1],datos[i,2],marker='*',c='y')
    else:
        ax.scatter(datos[i,0],datos[i,1],datos[i,2],marker='^',c='r')

ax.set_xlabel('Color')
ax.set_ylabel('Tasa aspecto')
ax.set_zlabel('Forma')

#Entrenamiento
A=np.zeros((4,4))
b=np.zeros((4,1))

for i in range(0,22):
    x=np.append([1],datos[i])
    x=x.reshape((4,1))
    y=etiquetas[i]
    A=A+x*x.T
    b=b+x*y
inversa=np.linalg.inv(A)
w=np.dot(inversa,b)
X = np.arange(0,1,0.1)
Y = np.arange(0,1,0.1)
X, Y =np.meshgrid(X,Y)
Z=-(w[0]+w[1]*X+w[2]*Y)/w[3]
surf = ax.plot_surface(X,Y,Z, cmap=cm.Reds)

#error de entrenamiento
prediccion=[]

for i in range(0,22):
    x=np.append([1],datos[i])
    x=x.reshape((4,1))
    prediccion.append(np.sign(np.dot(w.T,x)))
prediccion=np.array(prediccion).reshape((22))

efectividadEntrenamiento=(np.sum(prediccion==etiquetas)/22)*100
errorEntrenamiento =100-efectividadEntrenamiento

print("Efectividad entrenamiento "+str(efectividadEntrenamiento)+"%")
print("Error entrenamiento "+str(errorEntrenamiento)+"%")
    
#visualizar datos de prueba
datosPrueba=[]
etiquetasPrueba=[]
for i in range(1,5):
    datosPrueba.append(caracteristicas(cv2.imread("pruebaMosca_"+str(i)+".jpg")))
    etiquetasPrueba.append(1)
    datosPrueba.append(caracteristicas(cv2.imread("pruebaFalsa_"+str(i)+".jpg")))
    etiquetasPrueba.append(-1)
datosPrueba=np.array(datosPrueba)
etiquetasPrueba=np.array(etiquetasPrueba)

for i in range(0,8):
    if etiquetasPrueba[i]==1:
        ax.scatter(datosPrueba[i,0],datosPrueba[i,1],datosPrueba[i,2],marker='*',c='black')
    else:
        ax.scatter(datosPrueba[i,0],datosPrueba[i,1],datosPrueba[i,2],marker='^',c='blue')


#error prueba
prediccionPrueba=[]

for i in range(0,8):
    x=np.append([1],datosPrueba[i])
    x=x.reshape((4,1))
    prediccionPrueba.append(np.sign(np.dot(w.T,x)))
    
prediccionPrueba=np.array(prediccionPrueba).reshape((8))

efectividadPrueba=(np.sum(prediccionPrueba==etiquetasPrueba)/8)*100
errorPrueba =100-efectividadPrueba

print("Efectividad Prueba "+str(efectividadPrueba)+"%")
print("Error Prueba "+str(errorPrueba)+"%")

#prediccion unica imagen
imagen="pruebaMosca_5.jpg"
img=cv2.imread(imagen)
x=np.append([1],caracteristicas(img))
if np.sign(np.dot(w.T,x))==1:
    print(imagen+" Es una mosca")
else:
    print((imagen+" No es una mosca"))  