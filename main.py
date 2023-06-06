import cv2
from random import randrange
import numpy as np

def basic():
    img = cv2.imread('data/cat.jpg')
    #cv2.imshow('Image', img)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('Gray Image', gray)
    
    adaptiveThresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)
    #cv2.imshow('Adaptive Threshold Mean', adaptiveThresh)

    # adaptiveThresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 0)
    # cv2.imshow('Adaptive Threshold Gausian', adaptiveThresh)

    edges = cv2.Canny(gray, 50, 100)
    #cv2.imshow('Edges', edges)

    blur = cv2.blur(img, (11, 11))
    cv2.imshow('Blur', blur)

    blur = cv2.medianBlur(img, 11)
    #cv2.imshow('Median Blur', blur)

    blur = cv2.GaussianBlur(img, (13, 13), 0, 0)
    #cv2.imshow('Gaussian Blur', blur)

    cv2.waitKey()


def training_task_bananas():
    img = cv2.imread('data/bananas.jpg')
    #cv2.imshow('Image', img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Canny 
    #blur = cv2.blur(gray, (5, 5))
    blur = cv2.GaussianBlur(gray, (5, 5), 3)
    # cv2.imshow('Blur', blur)

    edges = cv2.Canny(blur, 20, 110)    
    # cv2.imshow('Edges', edges)

    # Zvacsi obrysy bananov
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 5))
    edges = cv2.dilate(edges, kernel, iterations=3)
    cv2.imshow('Edges', edges)

    # Zmeni farbu pozadia, aby sa dali rozlisit banany
    edges = cv2.bitwise_not(edges)
    _, b = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    edges[b > 0] = 0
    cv2.imshow('Edges2', edges)

    contours, h = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print('All contours: ', len(contours))
    n = 0
    for i, c in enumerate(contours):
        colour = (randrange(255), randrange(255), randrange(255))
        # colour = (255, 0, 0)
        if cv2.contourArea(c) > 200:
            cv2.drawContours(img, contours, i, colour, 3)
            n += 1

        # cv2.putText(img, i)
    cv2.imshow('Contours', img)
    print('Possible bananas: ', n)

    cv2.waitKey()


def training_task_oranges():
    img = cv2.imread('data/oranges.jpg')
    mask = img[:, :, 2]  # R 
    # cv2.imshow('Image', mask)

    _, edges = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)
    
    # Zvacsi priestor medzi pomarancami aby sa dali lepsie rozlisit
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(13, 13))
    edges = cv2.erode(edges, kernel, iterations=3)
    cv2.imshow('Edges', edges)
    
    contours, h = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print('All contours: ', len(contours))
    n = 0
    for i, c in enumerate(contours):
        colour = (randrange(255), randrange(255), randrange(255))
        # colour = (255, 0, 0)
        if cv2.contourArea(c) > 200:
            cv2.drawContours(img, contours, i, colour, 3)
            n += 1

        # cv2.putText(img, i)
    cv2.imshow('Contours', img)
    print('Possible oranges: ', n)

    cv2.waitKey()


if __name__ == '__main__':
    #basic()    
    training_task_bananas()
    #training_task_oranges()
