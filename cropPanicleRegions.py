import sys
import os, argparse
import shutil
from glob import glob
from skimage import filters
from skimage import color
from skimage.io import imread, imsave

from skimage.measure import label, regionprops
from scipy import ndimage

path = "D:/data/2020_wheatPanicle_2dScan/穗轴扫描3/raw"
out_path = "D:/data/2020_wheatPanicle_2dScan/穗轴扫描3/crop"

flist = sorted(glob(path +'/*.jpg'))

for id in range(len(flist)):
    
    image = imread(flist[id])

    file_name = os.path.splitext(flist[id])[0].split('\\')[-1]
    filename = os.path.basename(flist[id])
    out_sub_dir = os.path.join(out_path, filename[:-4])
    if not os.path.isdir(out_sub_dir):
        os.makedirs(out_sub_dir)

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
            out_file_path = os.path.join(out_sub_dir, '{}_{}.png'.format(filename[:-4], i))
            #Bounding box (min_row, min_col, max_row, max_col).
            imsave(out_file_path, image[max(region.bbox[0]-50,0):min(region.bbox[2]+50, image.shape[0]), \
                max(region.bbox[1]-50,0):min(region.bbox[3]+50, image.shape[1])])
            i = i+1