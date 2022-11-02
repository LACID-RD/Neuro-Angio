import SimpleITK as sitk
import numpy as np 
import pydicom
from myshow import *
import tkinter as tk
from tkinter import filedialog
import os


def read_dicom(lower, upper):
    path = tk.Tk()
    path.withdraw()
    file_name = filedialog.askopenfilenames(title="Cardio")
    for i in file_name:
        path = os.path.abspath(i)
        dcm = pydicom.dcmread(i)
        arr = dcm.pixel_array
        img = sitk.GetImageFromArray(arr)
        binImg = sitk_threshold(arr, lower=lower, upper=upper)
        binArr = sitk.GetArrayFromImage(binImg)
        
        
        erosionImg = dilate_filter(binArr)
        erosionArr = sitk.GetArrayFromImage(erosionImg)
        binContourImg = binary_contour(erosionArr)
        
        
        myshow(binImg)
        myshow(binContourImg)
        myshow(erosionImg)


def sitk_threshold(arr, lower, upper, outside=0, inside=255):
    
    img = sitk.GetImageFromArray(arr)
    
    binT = sitk.BinaryThresholdImageFilter()
    
    binT.SetLowerThreshold(lower)
    binT.SetUpperThreshold(upper)
    
    binT.SetOutsideValue(outside)
    binT.SetInsideValue(inside)
    
    binImg = binT.Execute(img)
    
    return binImg


def binary_contour(arr):
    
    img = sitk.GetImageFromArray(arr)
    
    contour = sitk.BinaryContourImageFilter()
    contour.FullyConnectedOn()
    
    contour.SetBackgroundValue(0)
    contour.SetForegroundValue(255)
    
    contImg = contour.Execute(img)
    return contImg

def dilate_filter(arr):
    img = sitk.GetImageFromArray(arr)
    #erode = sitk.BinaryErodeImageFilter()
    dilate = sitk.BinaryDilateImageFilter()
    dilate.SetForegroundValue(255)  # Intensity value to erode
    #erode.SetBackgroundValue(0)  # Replacement value for eroded voxels
    
    dilate.SetKernelType(sitk.sitkBall)
    '''
    sitkBall
    sitkBox
    sitkCross
    '''
    dilate.SetKernelRadius(2)
    
    erosionImg = dilate.Execute(img)
    return erosionImg


read_dicom(700, 3000)