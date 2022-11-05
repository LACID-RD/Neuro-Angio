
from base64 import decode
from ntpath import join
from os import access
import glob
from turtle import width
from unittest import skip
import SimpleITK as sitk
import numpy as np
import pydicom as dicom
import cv2 as cv
import pydicom
import tkinter as tk
from tkinter import N, Image, filedialog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import operator
from collections import Counter
import pandas as pd
import time
from itertools import groupby
import gdcm
import sys
import os
from pydicom.data import get_testdata_files
from Class_VelFld import VelFld
from myshow import *
# from test import *

"""
###########################################
############# CLASES ######################
###########################################

class VelFld:
    def __init__ (self, slice, time,
    imgposition, triggertime,
    vx, vy, vz, accX, accY, accZ, fc):

        self.slice = slice
        self.time = time
        self.imgposition = imgposition
        self.triggertime = triggertime
        '''Atributos especificos de velocidades'''
        self.Vx = vx
        self.Vy = vy
        self.Vz = vz
        self.accX = accX
        self.accY = accY
        self.accZ = accZ
        self.fc = fc
"""


def DescomDicom(PathAux, inp, outp):

    reader = gdcm.ImageReader()
    # reader.SetFileName(self.path_name)
    PAI = os.path.join(PathAux, inp)
    PAO = os.path.join(PathAux, outp)
    # print(PAI)
    # print(PAO)
    reader.SetFileName(PAI)
    if not reader.Read():
        print("Error: No se leen correctamente las imágenes.")
        sys.exit(0)
    change = gdcm.ImageChangeTransferSyntax()
    change.SetTransferSyntax(gdcm.TransferSyntax(
        gdcm.TransferSyntax.ImplicitVRLittleEndian))
    change.SetInput(reader.GetImage())
    if not change.Change():
        print("Error: No se descomprime correctamente las imágenes.")
        sys.exit(0)
    writer = gdcm.ImageWriter()
    writer.SetFileName(PAO)
    writer.SetFile(reader.GetFile())
    writer.SetImage(change.GetOutput())
    if not writer.Write():
        print("Error: No se escriben correctamente las imágenes.")
        sys.exit(0)


def position_ROI(Dcm_im_ROI):
    '''
    Give the general position of the ROI used to be copy on every image
    '''
    Dcm_im8 = (Dcm_im_ROI).astype(np.uint8)
    showCrosshair = False
    r = cv.selectROI("Seleccione el ROI a analizar", Dcm_im8, showCrosshair)
    cv.destroyAllWindows()
    # r = top x, top y, bottom x, bottom y
    return r

###########################################
############# Threshold ###################
###########################################

def FunThres(arr,valThr):

    newArr=np.where(arr < valThr,0,arr)
    maskArr=np.where(newArr >= valThr,255,newArr)
    return maskArr

###########################################
############### Dilatar ###################
###########################################

def Dilate(binArr,dilRad):
    img=sitk.GetImageFromArray(binArr)
    dilate=sitk.BinaryDilateImageFilter()
    dilate.SetForegroundValue(255)
    dilate.SetKernelType(sitk.sitkBall)
    dilate.SetKernelRadius(dilRad)
    dilImg=dilate.Execute(img)
    dilArr=sitk.GetArrayFromImage(dilImg)
    return dilArr

###########################################
############# DEFINICION DE VARIABLES #####
###########################################

image_list, image_slices, pmesh, img_qtt, countmesh, result, dcm_im_all, ph, dcm_all, PDP = [
], [], [], 0, 0, 0, [], 0, [], []

###########################################
################# MAIN ####################
###########################################

# Carga las imágenes, genera lista de slice y genera objetos
def main():
    valThr=100
    counterbr=0

    path = tk.Tk()
    path.withdraw()
    file_name = filedialog.askdirectory(title="PPS")
    # Se lee la primera imagen
    for root, dirs, files in os.walk(file_name):
        for i in files:
            if counterbr != 0:
                continue
            if i.endswith(".dcm"):
                PathAux1 = os.path.join(root, i)
            counterbr += 1
    # print(PathAux1)

    # Extrae info DICOM de la primera imagen
    Dcm1 = pydicom.dcmread(PathAux1)
    nslices = Dcm1[0x2001, 0x1018].value
    nphases = Dcm1[0x2001, 0x1017].value
    height, width = (Dcm1.pixel_array).shape
    print("Data estudio, phase:",nphases,nslices,":nslices, matrix:",height,width)

    #Se generan objetos de la clase VelFld
    for n in range(nslices):
        image_list = []
        for j in range(nphases):
            image_list.append('imagen' + str(j))  # Lista de imágenes
            image_list[j] = VelFld(n+1, j+1)
        PDP.append(image_list)

    #Recorre todas las imagenes para llenar objetos
    counter = 0
    counter1 = 0
    for root,dirs,files in os.walk(file_name):
        counter1+=1
        for i in files:
            if i.endswith(".dcm"):
                counter += 1
                #Path's para tema descompresion
                out = "nuevo" + str(counter) + ".dcm"
                PAux = os.path.join(root, i)
                DescomDicom(root, i, out)
                Dcm = pydicom.dcmread(PAux)
                DcmDecompressed = pydicom.dcmread(os.path.join(root, out))
                #Se toman solo las imagenes de phase
                if str(Dcm[0x2005, 0x116e].value) == 'PCA':
                    CurrSlice = int(Dcm[0x2001, 0x100a].value)-1
                    CurrPhase = int(Dcm[0x2001, 0x1008].value)-1
                    PDP[CurrSlice][CurrPhase].imgposition = Dcm[0x0020, 0x0032].value
                    PDP[CurrSlice][CurrPhase].triggertime = Dcm[0x0018, 0x1060].value
                    #Imagen neta o resultante
                    if Dcm[0x2001, 0x101a].value[0] > 50 and Dcm[0x2001, 0x101a].value[1] > 50 and Dcm[0x2001, 0x101a].value[2] > 50:
                        PDP[CurrSlice][CurrPhase].Vr = DcmDecompressed.pixel_array
                        # img = FunThres(
                        #     PDP[CurrSlice][CurrPhase].Vr)
                        # PDP[CurrSlice][CurrPhase].Vr = sitk.GetArrayFromImage(img)
                        # PDP[CurrSlice][CurrPhase].Vr = img
                    #Componente RL o X    
                    elif Dcm[0x2001, 0x101a].value[0] > 50 and Dcm[0x2001, 0x101a].value[1] == 0 and Dcm[0x2001, 0x101a].value[2] == 0:
                        PDP[CurrSlice][CurrPhase].Vx = DcmDecompressed.pixel_array
                        #imagenbb = PDP[CurrSlice][CurrPhase].Vx
                        #with open("mat.npy", "wb") as f:
                        #np.save(f, np.array(PDP[int(nslices/2)][10].Vx))
                        #plt.hist(PDP[int(nslices/2)][10].Vx, bins=bin)
                        #plt.show()
                        # print(type(imagenbb))
                        # img = FunThres(imagenbb)
                        # img == imagenbb
                        #plt.imshow(imagenbb)
                        #plt.show()
                        # PDP[CurrSlice][CurrPhase].Vx = sitk.GetArrayFromImage(img)
                        # PDP[CurrSlice][CurrPhase].Vx = img
                    #Componente AP o Y
                    elif Dcm[0x2001, 0x101a].value[0] == 0 and Dcm[0x2001, 0x101a].value[1] > 50 and Dcm[0x2001, 0x101a].value[2] == 0:
                        PDP[CurrSlice][CurrPhase].Vy = DcmDecompressed.pixel_array
                        # img = sitk_threshold(
                        #     PDP[CurrSlice][CurrPhase].Vy, lower=150, upper=8000)
                        # PDP[CurrSlice][CurrPhase].Vy = sitk.GetArrayFromImage(img)
                        # PDP[CurrSlice][CurrPhase].Vy = img
                    #Componente FH o Z
                    elif Dcm[0x2001, 0x101a].value[0] == 0 and Dcm[0x2001, 0x101a].value[1] == 0 and Dcm[0x2001, 0x101a].value[2] > 50:
                        PDP[CurrSlice][CurrPhase].Vz = DcmDecompressed.pixel_array
                        # img = sitk_threshold(
                        #     PDP[CurrSlice][CurrPhase].Vz, lower=150, upper=8000)
                        # PDP[CurrSlice][CurrPhase].Vz = sitk.GetArrayFromImage(img)
                        # PDP[CurrSlice][CurrPhase].Vz = img
    print("Termina de llenar velocidades y calcula mascara")

    #Dilata mascara binaria dentro de los objetos
    for i in range(nslices):
        for j in range(nphases):
            binArr=FunThres(PDP[i][j].Vr,valThr)
            PDP[i][j].mask=Dilate(binArr,1)
            #Energia cinetica
            PDP[i][j].kine=np.divide(np.multiply(np.power(PDP[i][j].Vr,2),PDP[i][j].mask),255)
            #EnerMatrix = np.multiply(PDP[n][j].Vx, PDP[n][j].Vx) + np.multiply(PDP[n][j].Vy, PDP[n][j].Vy) + np.multiply(PDP[n][j].Vz, PDP[n][j].Vz)


    MatVr=PDP[int(nslices/2)][int(nphases/2)].Vr
    plt.imshow(MatVr)
    plt.title('MatVr')
    plt.show()
    DilVr=PDP[int(nslices/2)][int(nphases/2)].mask
    plt.imshow(DilVr)
    plt.title('Mask-dilate')
    plt.show()
    KinVr=PDP[int(nslices/2)][int(nphases/2)].kine
    plt.imshow(KinVr)
    plt.title('Kinetic')
    plt.show()
    #por si al DAni se le olvida como hacer un hist
    #HisVr=np.hstack(MatVr)
    #plt.hist(HisVr,bins="auto")
    #plt.title('Hist-Vr')
    #plt.show()
    
    #Calculo de la aceleracion
    """ACELERACION Y ENERGIA"""

    # accMatrixX = np.zeros(shape=(int(height),int(width)))
    # accMatrixY = np.zeros(shape=(int(height),int(width)))
    # accMatrixZ = np.zeros(shape=(int(height),int(width)))

    # EnerMatrix = np.zeros(shape=(int(height),int(width)))


    # for n in range(nslices):
    #     for j in range(nphases):
    #     #     if j == 0:
    #     #         continue
    #     #     deltaT = (float(PDP[n][j].triggertime) - float(PDP[n][j-1].triggertime))
    #     # # print(deltaT)
    #     #     accMatrixX = (PDP[n][j].Vx - PDP[n][j-1].Vx)/deltaT
    #     #     accMatrixY = (PDP[n][j].Vy - PDP[n][j-1].Vy)/deltaT
    #     #     accMatrixZ = (PDP[n][j].Vz - PDP[n][j-1].Vz)/deltaT
    #     #     EnerMatrix = np.multiply(PDP[n][j].Vx, PDP[n][j].Vx) + np.multiply(PDP[n][j].Vy, PDP[n][j].Vy) + np.multiply(PDP[n][j].Vz, PDP[n][j].Vz)
    #     #     PDP[n][j].fc = EnerMatrix
    #     #     PDP[n][j].accX = accMatrixX
    #     #     PDP[n][j].accY = accMatrixY
    #     #     PDP[n][j].accZ = accMatrixZ
    #         plt.imshow(PDP[n][j].Vr, cmap="bone")
    #         plt.imshow(PDP[n][j].Vx, cmap="bone")
    #         plt.imshow(PDP[n][j].Vy, cmap="bone")
    #         plt.imshow(PDP[n][j].Vz, cmap="bone")

    #         # plt.imshow(PDP[n][j].fc, cmap="rainbow")
    #         # plt.imshow(PDP[n][j].accX, cmap="rainbow")
    #         plt.show()

    # plt.imshow(PDP[2][2].fc)
    # """ENERGIA"""

    # hola = PDP[1][1].Vx
    # #print(hola)

    # pato= os.path.join(root,"")
    # pato=str(pato)+"/nuevo*"
    # print(pato)

    # for filename in glob.glob(pato):
    #     os.remove(filename)

    # """"Calculo del ROTACIONAL"""
    # #hacer el meshgrid (que nos da la posición donde se colocan los vectores)

    # rangex = abs(PDP[2][2].imgposition[0])*2
    # rangey = abs(PDP[2][2].imgposition[1])*2
    # rangez = abs(PDP[nslices-1][2].imgposition[2])*2

    # x=np.linspace(0,rangex,width)
    # y=np.linspace(0,rangey,height)
    # z=np.linspace(0,rangez,nslices)

    # X,Y,Z = np.meshgrid(x,y,z)

    # u=np.zeros(shape=(width,height,nslices))
    # v=np.zeros(shape=(width,height,nslices))
    # w=np.zeros(shape=(width,height,nslices))

    # for i in range(nslices):
    #     u[:,:,i]=PDP[i][2].Vx
    #     v[:,:,i]=PDP[i][2].Vy
    #     w[:,:,i]=PDP[i][2].Vz

    # print(u.shape,X.shape,u[0,0,0],type(u[0,0,0]) #ver si u[0,0,0] no es un numero entonces esta guardando PDP 240*240 veces

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # plt.quiver(X,Y,Z,u,v,w,length=0.000001 ,color='red')
    # plt.show()
    # dx,dy,dz = (0.83,0.83,1)

    # def curl_custom(u,v,w,dx,dy,dz):
    #     dwdy=np.gradient(w,dy,axis=0)
    #     dvdz=np.gradient(v,dz,axis=2)

    #     dudz=np.gradient(u,dz,axis=2)
    #     dwdx=np.gradient(w,dx,axis=1)

    #     dvdx=np.gradient(v,dx,axis=1)
    #     dudy=np.gradient(u,dy,axis=0)

    #     return [dwdy-dvdz, dudz-dwdx, dvdx-dudy]
    # [wx,wy,wz]=curl_custom(u,v,w,dx,dy,dz)
    # #print(wx,wy,wz,sep='\n \n')

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # plt.quiver(X,Y,Z,wx,wy,wz,length=0.000001,color='red')
    # plt.show()


if __name__ == '__main__':
    main()

# for root,dirs,files in os.walk(file_name):

#     for i in files:
#         if i.endswith(".dcm"):
#             PathAux=os.path.join(root,i)
#             DescomDicom(PathAux)
#             #Dcm=pydicom.dcmread(PathAux)
#             Dcm=pydicom.dcmread(PathAux)
#             # for i in file_name:
#             # Dcm = pydicom.dcmread(i)
#             if str(Dcm[0x2005, 0x116e].value) == 'PCA':
#                 for j in range(nslices):
#                     for k in range(nphases):
#                         # Filtrado según componentes
#                         if int(Dcm[0x2001, 0x100a].value) is int(j+1) and int(Dcm[0x2001,0x1008].value) is int(k+1):
#                             setattr(PDP[j][k], "imgposition", Dcm[0x0020,0x0032].value)
#                             setattr(PDP[j][k], "triggertime", Dcm[0x0018,0x1060].value)
#                             if Dcm[0x2001, 0x101a].value[0]>50 and Dcm[0x2001, 0x101a].value[1]>50 and Dcm[0x2001, 0x101a].value[2]>50: # Imagen neta
#                                 # PDP[j][k].vx = Dcm.pixel_array
#                                 continue
#                             if Dcm[0x2001, 0x101a].value[0]>50 and Dcm[0x2001, 0x101a].value[1]==0 and Dcm[0x2001, 0x101a].value[2]==0: # RL # X
#                                 setattr(PDP[j][k], "Vx", Dcm.pixel_array)
#                             elif Dcm[0x2001, 0x101a].value[0]==0 and Dcm[0x2001, 0x101a].value[1]>50 and Dcm[0x2001, 0x101a].value[2]==0: # AP # Y
#                                 setattr(PDP[j][k], "Vy", Dcm.pixel_array)
#                             elif Dcm[0x2001, 0x101a].value[0]==0 and Dcm[0x2001, 0x101a].value[1]==0 and Dcm[0x2001, 0x101a].value[2]>50: # FH # Z
#                                 setattr(PDP[j][k], "Vz", Dcm.pixel_array)

# hola = PDP[1][1]


# '''Generamos las instancias de la clase Image utilizando lectura de los dicoms.'''

# ndeah = 0

# # print(file_name[0],type(file_name))
# Dcm1 = pydicom.dcmread(file_name[0])

# nslices = Dcm1[0x2001, 0x1018].value
# nphases = Dcm1[0x2001, 0x1017].value
# height, width =  (Dcm1.pixel_array).shape

# print(nslices,nphases)
# PDP = []
# #print(PDP)

# for n in range(nslices):
#     image_list = []
#     for j in range(nphases):
#         image_list.append('imagen' + str(j)) # Lista de imágenes
#         image_list[j] = VelFld (n+1, j+1, None, None, None, None, None, None, None, None, None)
#         # ndeah = ndeah+ 1
#     PDP.append(image_list)

# # for i

# # print((PDP[1][1].slice))

# for i in file_name:
#     Dcm = pydicom.dcmread(i)
#     if str(Dcm[0x2005, 0x116e].value) == 'PCA':
#         for j in range(nslices):
#             for k in range(nphases):
#                 # Filtrado según componentes
#                 if int(Dcm[0x2001, 0x100a].value) is int(j+1) and int(Dcm[0x2001,0x1008].value) is int(k+1):
#                     setattr(PDP[j][k], "imgposition", Dcm[0x0020,0x0032].value)
#                     setattr(PDP[j][k], "triggertime", Dcm[0x0018,0x1060].value)
#                     if Dcm[0x2001, 0x101a].value[0]>50 and Dcm[0x2001, 0x101a].value[1]>50 and Dcm[0x2001, 0x101a].value[2]>50: # Imagen neta
#                         # PDP[j][k].vx = Dcm.pixel_array
#                         continue
#                     if Dcm[0x2001, 0x101a].value[0]>50 and Dcm[0x2001, 0x101a].value[1]==0 and Dcm[0x2001, 0x101a].value[2]==0: # RL # X
#                         setattr(PDP[j][k], "Vx", Dcm.pixel_array)
#                     elif Dcm[0x2001, 0x101a].value[0]==0 and Dcm[0x2001, 0x101a].value[1]>50 and Dcm[0x2001, 0x101a].value[2]==0: # AP # Y
#                         setattr(PDP[j][k], "Vy", Dcm.pixel_array)
#                     elif Dcm[0x2001, 0x101a].value[0]==0 and Dcm[0x2001, 0x101a].value[1]==0 and Dcm[0x2001, 0x101a].value[2]>50: # FH # Z
#                         setattr(PDP[j][k], "Vz", Dcm.pixel_array)

# hola = PDP[1][1]
# #print(hola.Vx)

# """FILTRADO VER CON RODRI"""

# """ACELERACION"""

# accMatrixX = np.zeros(shape=(int(height),int(width)))
# accMatrixY = np.zeros(shape=(int(height),int(width)))
# accMatrixZ = np.zeros(shape=(int(height),int(width)))

# forceMatrix = np.zeros(shape=(int(height),int(width)))


# for n in range(nslices):
#     for j in range(nphases):
#         if j == 0:
#             continue
#         deltaT = (float(PDP[n][j].triggertime) - float(PDP[n][j-1].triggertime))
#     # print(deltaT)
#         accMatrixX = (PDP[n][j].Vx - PDP[n][j-1].Vx)/deltaT
#         accMatrixY = (PDP[n][j].Vy - PDP[n][j-1].Vy)/deltaT
#         accMatrixZ = (PDP[n][j].Vz - PDP[n][j-1].Vz)/deltaT
#         forceMatrix = np.multiply(PDP[n][j].Vx, PDP[n][j].Vx) + np.multiply(PDP[n][j].Vy, PDP[n][j].Vy) + np.multiply(PDP[n][j].Vz, PDP[n][j].Vz)
#         setattr(PDP[n][j], "fc", forceMatrix)
#         setattr(PDP[n][j], "accX", accMatrixX)
#         setattr(PDP[n][j], "accY", accMatrixY)
#         setattr(PDP[n][j], "accZ", accMatrixZ)

# plt.imshow(PDP[2][2].fc)
# """ENERGIA"""


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
