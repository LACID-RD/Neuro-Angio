
import SimpleITK as sitk
import numpy as np
import pydicom as dicom
import cv2 as cv
import pydicom
import tkinter as tk
from tkinter import Image, filedialog
import matplotlib.pyplot as plt
import numpy as np
import operator 
from collections import Counter
import pandas as pd
import time
from itertools import groupby

###########################################
############# CLASES ######################
###########################################

class VelFld:
    def __init__ (self, slice, time, info, 
    imgposition, pxspacingx, pxspacingy,
    vx, vy, vz):

        self.slice = slice
        self.time = time
        self.info = info
        self.dir = dir 
        self.imgposition = imgposition
        self.pxspacingx = pxspacingx
        self.pxspacingy = pxspacingy
        '''Atributos especificos de velocidades'''   
        self.Vx = vx 
        self.Vy = vy
        self.Vz = vz


def position_ROI(Dcm_im_ROI):
    '''
    Give the general position of the ROI used to be copy on every image
    '''
    Dcm_im8 = (Dcm_im_ROI).astype(np.uint8)
    showCrosshair = False
    r = cv.selectROI("Seleccione el ROI a analizar", Dcm_im8,showCrosshair)
    cv.destroyAllWindows()
    # r = top x, top y, bottom x, bottom y
    return r

###########################################
############# DEFINICION DE VARIABLES #####
###########################################

image_list, image_slices, pmesh, img_qtt, countmesh, result, dcm_im_all, ph, dcm_all = [], [], [], 0, 0, 0, [], 0, []

###########################################
################# MAIN ####################
###########################################


# Carga las imágenes, genera lista de slice y genera objetos
path = tk.Tk()
path.withdraw()
file_name = filedialog.askopenfilenames()

'''Generamos las instancias de la clase Image utilizando lectura de los dicoms.'''

for i in file_name:
    Dcm = pydicom.dcmread(i)
    # Filtrado de imágenes de magnitud
    if str(Dcm[0x2005, 0x116e].value) == 'PCA': 
        image_list.append('imagen' + str(img_qtt)) # Lista de imágenes
        image_slices.append(Dcm[0x2001,0x100a].value)
        dcm_all.append(Dcm)
        #print(type(Dcm))
#        dcm_im_all.append(Dcm.pixel_array)
    #ph = Dcm[0x0018,0x9232].value

# define a function for key
def key_func(k):
    return k[0x0018,0x1060]

# define a function for key
def key_func1(k):
    return k[0x2001,0x100a]

# sort INFO data by 'company' key.
dcm_all = sorted(dcm_all, key=key_func)
print(dcm_all)

# key_func = lambda x: x[0x0018,0x1060].value
# key_func1 = lambda x: x[0x2001,0x100a].value

# groupDcm = groupby(dcm_all, key_func(dcm_all))

# grouppedByTrigger = []
# for key, group in groupDcm:
#     grouppedByTrigger.append(list(group))

# for i in range(len(grouppedByTrigger)):
#     groupDcmFinal = groupby(grouppedByTrigger[i], key_func1)

#     grouppedByTSlice = []
#     for key, group in groupDcmFinal:
#         grouppedByTSlice.append(list(group))


for j in range(len(grouppedByTrigger)):
    for k in range(len(grouppedByTrigger[j])):
        dir = 0
        Dcm = grouppedByTrigger[j][k]
        # Filtrado según componentes
        if Dcm[0x2001, 0x101a].value[0]>50 and Dcm[0x2001, 0x101a].value[1]>50 and Dcm[0x2001, 0x101a].value[2]>50: # Imagen neta
            dir = 1
        elif Dcm[0x2001, 0x101a].value[0]>50 and Dcm[0x2001, 0x101a].value[1]==0 and Dcm[0x2001, 0x101a].value[2]==0: # RL # X
            dir = 2
            imgvx = np.copy(Dcm.pixel_array) 
        elif Dcm[0x2001, 0x101a].value[0]==0 and Dcm[0x2001, 0x101a].value[1]>50 and Dcm[0x2001, 0x101a].value[2]==0: # AP # Y
            dir = 3
            imgvy = np.copy(Dcm.pixel_array) 
        elif Dcm[0x2001, 0x101a].value[0]==0 and Dcm[0x2001, 0x101a].value[1]==0 and Dcm[0x2001, 0x101a].value[2]>50: # FH # Z
            dir = 4
            imgvz = np.copy(Dcm.pixel_array)

#     # ARMAR UN SOLO OBJ CON LAS IMAGENES DE LAS 3 COMPONENTES COMO 3 ATRIBUTOS DISTINTOS
    image_list[img_qtt] = VelFld(Dcm[0x2001,0x100a].value, Dcm[0x0018,0x1060].value, 
    np.copy(Dcm.pixel_array), Dcm, Dcm[0x0020,0x0032].value, Dcm[0x0028,0x0030].value[0], 
    Dcm[0x0028,0x0030].value[1], imgvx, imgvy, imgvz) # Lista de objetos tipo image
    np.copy(Dcm.pixel_array), Dcm, Dcm[0x0020,0x0032].value, Dcm[0x0028,0x0030].value[0], 
    dcm_im_all.append(Dcm.pixel_array)
    img_qtt = img_qtt + 1    

# groupby(dcm)
#         ''' Para crear las instancias de la clase definiria primerp cada tag 
#         Ej Imagen(slice, tiempo, ..., vr, vx, vy, vz)
#         '''
#         #slice = 

#         '''Tag velocidades: (0x2001, 0x101a)'''

#         vx = ...
#         vy = ...
#         vz = ...

#         dir = 0
#         # Filtrado según componentes
#         if Dcm[0x2001, 0x101a].value[0]>50 and Dcm[0x2001, 0x101a].value[1]>50 and Dcm[0x2001, 0x101a].value[2]>50: # Imagen neta
#             dir = 1
#             imgvr = np.copy(Dcm.pixel_array)

#         elif Dcm[0x2001, 0x101a].value[0]>50 and Dcm[0x2001, 0x101a].value[1]==0 and Dcm[0x2001, 0x101a].value[2]==0: # RL # X
#             dir = 2
#             imgvx = np.copy(Dcm.pixel_array) 

#         elif Dcm[0x2001, 0x101a].value[0]==0 and Dcm[0x2001, 0x101a].value[1]>50 and Dcm[0x2001, 0x101a].value[2]==0: # AP # Y
#             dir = 3
#             imgvy = np.copy(Dcm.pixel_array) 

#         elif Dcm[0x2001, 0x101a].value[0]==0 and Dcm[0x2001, 0x101a].value[1]==0 and Dcm[0x2001, 0x101a].value[2]>50: # FH # Z
#             dir = 4
#             imgvz = np.copy(Dcm.pixel_array) 

#         # ARMAR UN SOLO OBJ CON LAS IMAGENES DE LAS 3 COMPONENTES COMO 3 ATRIBUTOS DISTINTOS
#         image_list[img_qtt] = Imagen(Dcm[0x2001,0x100a].value, Dcm[0x0018,0x1060].value, 
#         np.copy(Dcm.pixel_array), Dcm, Dcm[0x0020,0x0032].value, Dcm[0x0028,0x0030].value[0], 
#         Dcm[0x0028,0x0030].value[1], vx, vy, vz) # Lista de objetos tipo image
#         np.copy(Dcm.pixel_array), Dcm, Dcm[0x0020,0x0032].value, Dcm[0x0028,0x0030].value[0], 
#         dcm_im_all.append(Dcm.pixel_array)
#         img_qtt = img_qtt + 1

# # Ordena los objetos y calcula variables
# sorted_image_list  = sorted(image_list, key = operator.attrgetter('dir', 'slice', 'time')) 
# imgslctr = Counter(image_slices) # Número de slice : Cantidad de imágenes
# slice_qtt = max(imgslctr.keys()) # Cantidad de slices
# sl_n_imgs = int(max(imgslctr.values())/4) # Cantidad de imágenes por Slice
# img_by_type = int(img_qtt/4)
# height, width = (image_list[0].img).shape # Tamaño de las imágenes

# # Matrices a trabajar 
# maskmatrix = np.zeros((slice_qtt,height,width), np.uint16)    # Máscara de todos los slice  
# maskimage = np.zeros((height,width), np.uint16)               # Máscara unitaria
# maskimageROI = np.zeros((height,width), np.uint64)            # Máscara unitaria para ROI
# maskimgmatrix = np.zeros((slice_qtt,height,width), np.uint16) # Máscara pos-ROI de todos los slice
# reconstucted_matrix = np.zeros((height,width), np.uint16)


# '''Generamos el treshold'''


# # Genera el Threshold y plasma en un arreglo toda la info (en ese arreglo se va a realizar el ROI)
# # Verifica que exista info en el slice anterior para eliminar vasos o info no deseada
# for q in range(slice_qtt): 
#     img_org = sorted_image_list[q * sl_n_imgs].img 
#     prev_img_org = sorted_image_list[(q-1) * sl_n_imgs].img
#     for j in range(height): 
#         for k in range(width):
#             if img_org[j,k] <= 80: # 500 para FFE
#                 maskimage[j,k] = 0
#                 maskimageROI[j,k] = maskimageROI[j,k] + 0
#             else:
#                 if q == 0:
#                     maskimage[j,k]= 255
#                     maskimageROI[j,k] = maskimageROI[j,k] + 128
#                 else:
#                     # Comparacion con el slice anterior (para todos menos para q=0)
#                     for o in range(3):
#                         for p in range(3):
#                             if prev_img_org[j+o-1, k+p-1] == 255:
#                                 result = result + 1
#                             else:
#                                 result = result
#                     if result>0:
#                         maskimage[j+o-1, k+p-1] = 255
#                         maskimageROI[j+o-1, k+p-1] = maskimageROI[j+o-1, k+p-1] + 128
#                     else:
#                         maskimage[j+o-1, k+p-1] = 0
#                         maskimageROI[j+o-1, k+p-1] = maskimageROI[j+o-1, k+p-1]
    
# # Se realiza el ROI y se elimina la info fuera de él esto es para el slice maximo (me quedaba fuera de rango)
#     if q == (slice_qtt-1):
#         utROI = position_ROI(maskimageROI)
#         for l in range(width):
#             for m in range(height):
#                 if m>float(utROI[0]) and m<float(utROI[0]+utROI[2]) and l>float(utROI[1]) and l<float(utROI[1]+utROI[3]):
#                     maskimage[l,m]=maskimage[l,m]
#                 else:
#                     maskimage[l,m] = 0 
#     maskimgmatrix[q] = maskimage

# # hago lo mismo que hacia en las lineas anteriores pero para los demas slices

# for q in range(slice_qtt):
#     for l in range(width):
#             for m in range(height):
#                 if m>float(utROI[0]) and m<float(utROI[0]+utROI[2]) and l>float(utROI[1]) and l<float(utROI[1]+utROI[3]):
#                     maskimgmatrix[q,l,m]=maskimgmatrix[q,l,m]
#                 else:
#                     maskimgmatrix[q,l,m] = 0

# ####### ACA VER DE USAR SIMPLE ITK SI LOS GRADIENTES SON BIDIRECCIONALES

# #####ACA PAU FIJARME DE VER ESTO PARA ARMAR EL MESH CON BOUNDARY CELLS
# # Se realiza la deteccion de bordes con sobel
# for q in range(slice_qtt):
#     maskimageinv = cv.flip(maskimgmatrix[q],-1)
#     sobelx = cv.Sobel(maskimgmatrix[q],-1,1,0,ksize=1) # Image , DDepth = -1 (uint8) would be the result
#     sobely = cv.Sobel(maskimgmatrix[q],-1,0,1,ksize=1) # 1 , 0  for dx   and  0 , 1 for dy
#     sobeldir = sobelx + sobely
#     sobelxinv = cv.Sobel(maskimageinv,-1,1,0,ksize=1) 
#     sobelyinv = cv.Sobel(maskimageinv,-1,0,1,ksize=1)
#     sobelinv = sobelxinv + sobelyinv 
#     reconstucted_matrix = cv.flip(sobelinv,-1)
#     sobel = sobeldir +  reconstucted_matrix
#     maskmatrix[q] = sobel

#     # Seleccion de los valores internos a los bordes 
#     for i in range(height):
#         for j in range (width):
#             if (j > 1) and (i > 1): # FIJARME DE OPTIMIZARLO
#                 if (maskmatrix[q][i][j] == 0) and (maskmatrix[q][i-1][j] != 0) and (maskmatrix[q][i][j-1] != 0):
#                     maskmatrix[q][i][j] = 200 # 200 por poner un valor que sea exclusivo de los pixeles internos
    
# # print(maskmatrix)

# ######## ESTO QUEDARIA SOLUCIONADO CUANDO LO META COMO ATRIBUTO

# """REVISAR ESTO!!!!!!"""

# # # Generación de arreglo vectorial para cada punto y csv correspondiente
# # vector_array = np.ndarray((countmesh,3)) # Definir point_qtt
# # counter = 0 
# # for q in range(slice_qtt):
# #     # ESTA LINEA ES LO MAS NEFASTO DEL UNIVERSO PERO NECESITO SABER
# #     # if sorted_image_list[q * sl_n_imgs].imgposition[2] >= -44.19343 and sorted_image_list[q * sl_n_imgs].imgposition[2] <= -31.0219: #VALORES SACADOS DE PARAVIEW
# #     for j in range(height): 
# #         for k in range(width): 
# #             if k > float(utROI[1]) and k < float(utROI[1] + utROI[3]) and j > float(utROI[0]) and j < float(utROI[0] + utROI[2]): # Ajustar al Roi
# #                 if maskmatrix[q][k][j] > 0: 
# #                     vector_x = sorted_image_list[((q*sl_n_imgs)+img_by_type*1)].img[k,j]
# #                     vector_y = sorted_image_list[((q*sl_n_imgs)+img_by_type*2)].img[k,j]
# #                     vector_z = sorted_image_list[((q*sl_n_imgs)+img_by_type*3)].img[k,j]
# #                     vector_array[counter] = [vector_x, vector_y , vector_z]
# #                     counter += 1
# # print(vector_array)

# ####### REVISAR, CALCUL