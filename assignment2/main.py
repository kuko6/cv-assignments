import cv2
import numpy as np


def lab_conversion():
    img = cv2.imread('data/HMRegistred.png').astype(np.float32)
    img /= 255.0
    print(np.max(img))

    target = cv2.imread('data/HMRegistred_target.png').astype(np.float32)
    target /= 255.0

    imgLab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    cv2.imshow('Lab', imgLab)

    targetLab = cv2.cvtColor(target, cv2.COLOR_BGR2Lab)
    cv2.imshow('Target Lab', targetLab)

    print(np.max(targetLab[:,:,0]), np.min(targetLab[:,:,0]))
    print(np.max(targetLab[:,:,1]), np.min(targetLab[:,:,1]))
    print(np.max(targetLab[:,:,2]), np.min(targetLab[:,:,2]))

    target_mean = targetLab.mean(axis=0).mean(axis=0) # toto je [L2, A2, B2]
    print(target_mean)

    print(target_mean.shape)

def keypoints():
    # Load and convert them to gray scale
    img = cv2.imread('data/local_descriptors_task/lookup.tif', 0)
    patch1 = cv2.imread('data/local_descriptors_task/patch1.tif', 0)
    patch2 = cv2.imread('data/local_descriptors_task/patch2.tif', 0)
    patch3 = cv2.imread('data/local_descriptors_task/patch3.tif', 0)

    sift = cv2.SIFT_create()
    img_kp = sift.detect(img)
    image_with_keypoints = cv2.drawKeypoints(img, img_kp, None)

    cv2.imshow('Image with Keypoints', image_with_keypoints)
    

if __name__ == '__main__':
    # keypoints()
    # lab_conversion()
    img = cv2.imread('data/HMRegistred.png').astype(np.float32)
    img = img/255
    # cv2.imshow('gray', cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    # cv2.imshow('img', img)
    # ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # cv2.imshow('ycrcb', ycrcb[:, :, 2])
    # cv2.imshow('hsv', cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    # cv2.imshow('xyz', cv2.cvtColor(img, cv2.COLOR_BGR2XYZ))
    cv2.imshow('lab', cv2.cvtColor(img, cv2.COLOR_BGR2Lab))
    cv2.waitKey()
