import cv2
import numpy as np
import copy
import math


def super_pixels():
    # img = cv2.imread('data/segmentation/butterfly.jpeg')
    img = cv2.imread('data/segmentation/JPEGImages/2007_000033.jpg') # plane
    # img = cv2.imread('data/segmentation/JPEGImages/2007_000032.jpg') # plane
    # img = cv2.imread('data/segmentation/JPEGImages/2007_000068.jpg') # bird
    
    if img.shape[0] > 1000:
        img = cv2.resize(img, (img.shape[1]//10, img.shape[0]//10), interpolation=cv2.INTER_NEAREST)
    
    cv2.imshow('img', img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # inicialize SLIC
    slic = cv2.ximgproc.createSuperpixelSLIC(gray, cv2.ximgproc.SLIC, region_size=30, ruler=20)
    slic.iterate(20)

    mask = slic.getLabelContourMask(thick_line=True)
    contour_img = copy.deepcopy(img)
    contour_img[mask==255] = (0, 0, 255)
    cv2.imshow('contours', contour_img)

    # get corresponding labels for each pixel
    labels = slic.getLabels()
    # print(labels.shape)

    num_labels = len(np.unique(labels))
    # print(num_labels)

    # compute mean colour for each label (group of pixels)
    mean_colours = np.zeros((num_labels, 3), dtype=np.uint8)
    for label in np.unique(labels):
        tmp_mask = (labels == label).astype(np.uint8)
        mean_colours[label] = cv2.mean(img, mask=tmp_mask)[:3]
    
    # print(mean_colours.shape)
    # print(mean_colours)

    # create a colour image by assigning mean colours to labels
    mean_colour_img = mean_colours[labels]
    # print(mean_colour_img.shape)

    cv2.imshow('mean colours', mean_colour_img)
    # cv2.imshow('mask', mask)


if __name__ == '__main__':
    super_pixels()

    cv2.waitKey()
    cv2.destroyAllWindows()