import numpy as np
from numpy import ma, uint8
from numpy.core.numeric import ones_like
from scipy import ndimage
from skimage import measure
from skimage.io import imread, imsave

from skimage import filters
from skimage import color
from skimage.morphology.grey import dilation, erosion
from skimage.morphology.selem import square

from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops, regionprops_table
#from skimage.morphology import medial_axis, skeletonize
import matplotlib.pyplot as plt
from skimage import morphology
import os
from glob import glob
import math
import pandas as pd
import skfmm


def keepLargestComponent(bw):
    labels = label(bw)
    largestComponent = labels == np.argmax(np.bincount(labels.flat, weights = bw.flat))
    return largestComponent

def calStatTexture(data, index):
    val_names = ['Area', 'Length', 'Width', 'Width_Min'] 
    mean = np.mean(data)
    std = np.std(data)
    CV = std/mean
    skewness = pd.Series(data).skew()
    kurtosis = pd.Series(data).kurt()
    SM = 1 - 1/(1+std*std)
    return [val_names[index]+'_Mean', mean, val_names[index]+'_Std', std, \
            val_names[index]+'_Skewness', skewness, val_names[index]+'_Kurtosis', kurtosis, \
            val_names[index]+'_CV', CV, val_names[index]+'_SM', SM]

def measurePanicle(bw):
    bw = np.pad(bw, [[50,50], [50,50]], 'constant')
    distance = ndimage.distance_transform_edt(bw)
    coords = peak_local_max(distance, min_distance = 25)#, footprint=np.ones((5, 5)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True  

    gdt = ones_like(bw)
    gdt = ma.masked_array(gdt, bw==0)
    gdt[50, bw[50, :]>0] = 0
    gd = skfmm.distance(gdt)

    markers_gd = gd[np.nonzero(mask)] 
    markers_l = np.argsort(np.argsort(markers_gd)) + 1 

    markers = np.zeros(distance.shape, dtype=uint8)  
    markers[np.nonzero(mask)] = markers_l   

    #markers, _ = ndimage.label(mask)
    
    labels = watershed(-distance, markers, mask=bw)

    se = square(3)
    ero_image = labels.copy()
    ero_image[ero_image==0] = ero_image.max() + 1
    ero_image = erosion(ero_image, se)
    ero_image[labels==0] = 0
    grad = dilation(labels, se) - ero_image
    grad[labels==0] = 0
    grad[grad>0] = 1

    skel = morphology.medial_axis(bw)#morphology.skeletonize(bw)
    joint_points = filters.rank.sum(skel*1, se) * skel


    props = regionprops(labels, joint_points)
    labels_on_skel = labels * grad
    dist_on_skel = distance * grad
    props_on_skel = regionprops(labels_on_skel, dist_on_skel)

    solidity = np.array([prop.solidity for prop in props]) # solidity[index:len(solidty)]
    area = np.array([prop.area for prop in props])
    length = np.array([prop.major_axis_length for prop in props])
    width = np.array([prop.minor_axis_length for prop in props])
    width_min = 2*np.array([prop.max_intensity for prop in props_on_skel]) 
    joint_point = np.array([prop.max_intensity for prop in props]) 

    index = np.argmax(joint_point[1:len(joint_point)]>3)#np.argmax(solidity<0.945)
    if index==0:
        if solidity[0]>0.945:
            index = 1
    else:
        index = index + 1

    labels[labels<=index] = 0

    traits = []
    traits.append(['Total_Branch_Num', len(solidity)-index])
    traits.append(calStatTexture(area[index:len(area)], 0))
    traits.append(calStatTexture(length[index:len(length)], 1))
    traits.append(calStatTexture(width[index:len(width)], 2))
    traits.append(calStatTexture(width_min[index:len(width_min)], 3))

    for i in range(index, len(solidity)):
        label = i - index
        traits.append(['Branch{}_Area'.format(label), props[i].area])
        traits.append(['Branch{}_Length'.format(label), props[i].major_axis_length])
        traits.append(['Branch{}_Width'.format(label), props[i].minor_axis_length])
        traits.append(['Branch{}_Width_Min'.format(label), 2*props_on_skel[i].max_intensity])
    return labels, traits

path = "D:/data/2020赵县穗轴I"
out_path = "D:/data/2020赵县穗轴I"
csv_file = out_path + "/traits.csv"
#open('D:/data/2020赵县穗轴I/test.csv', 'a+')

flist = sorted(glob(path +'/*.jpg'))
file = open(csv_file, 'a+')

for id in range(len(flist)):
    
    image = imread(flist[id])
    file_name = os.path.splitext(flist[id])[0].split('\\')[-1]

    # convert the image to grayscale
    gray_image = color.rgb2gray(image)
    blurred_image = filters.gaussian(gray_image, sigma=1.0)
    thre = filters.threshold_otsu(blurred_image)
    bw_image = blurred_image < thre
    bw_image = ndimage.binary_fill_holes(bw_image)

    #bw_image = morphology.remove_small_objects(bw_image, 5000)
    label_image = label(bw_image)
    panicles = regionprops(label_image, blurred_image)
    i = 0

    for region in panicles:
        if region.area > 10000 and region.minor_axis_length > 20:
            out = []
            #sampleID = 'file_' + i
            sampleID = file_name+'_{}'.format(i)
            out.append([sampleID])
            seg_im, traits_arr = measurePanicle(region.image)
            out.append(traits_arr)
            s = [y for x in out for y in x]
            s = str(s).replace('[','').replace(']','')
            s = s.replace("'",'') +'\n' 
            file.write(s)
            seg_im = color.label2rgb(seg_im, bg_label = 0, bg_color = 'black')
            imsave(os.path.join(out_path, sampleID+"_label.png"), seg_im)
            ori_im = np.pad( region.intensity_image, [[50,50], [50,50]], 'constant')
            imsave(os.path.join(out_path, sampleID+"_gray.png"), ori_im)
            i = i+1

file.close() 




