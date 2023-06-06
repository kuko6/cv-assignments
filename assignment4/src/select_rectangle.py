import cv2
import copy
import numpy as np

img = None

point1 = None
point2 = None
rect = None

# modes
select_rectangle = True

# callback function for mouse events
# https://docs.opencv.org/3.4/db/d5b/tutorial_py_mouse_handling.html
def mouse_callback(event, x, y, flags, param):
    global img, point1, point2, rect, select_rectangle
    
    # left mouse button events
    if event == cv2.EVENT_LBUTTONDOWN:
        if select_rectangle:
            point1 = (x, y)
            img = cv2.circle(img, point1, 4, (0, 0, 255), -1)
        
    # right mouse button events
    elif event == cv2.EVENT_RBUTTONDOWN:
        if select_rectangle:
            point2 = (x, y)
            img = cv2.circle(img, point2, 4, (255, 0, 0), -1)

    # when both points are set, draw a rectangle connecting them
    if point1 and point2 and select_rectangle:
        rect = (min(point1[0], point2[0]), min(point1[1], point2[1]), abs(point1[0] - point2[0]), abs(point1[1] - point2[1]))
        cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
        select_rectangle = False
        

def crop_img(img, rect):
    if rect is None:
        rect = (0, 0, img.shape[1], img.shape[0])
        print(rect)
    cropped_img = img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
    return cropped_img


def main(image=None):
    global img, point1, point2, rect, select_rectangle

    if image is None:
        original_img = cv2.imread("data/A.png")
    else:
        original_img = image
    
    img = copy.deepcopy(original_img)
    cropped_img = None

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_callback)

    while True:
        k = cv2.waitKey(20)

        if k == ord('q'):
            break

        elif k == ord('r'):
            point1, point2, rect = None, None, None
            img = copy.deepcopy(original_img)
            select_rectangle = True

        elif k == ord('s'):
            cropped_img = crop_img(img, rect)
            cv2.imshow('cropped', cropped_img)

        elif k == ord('d') and cropped_img is not None:
            cv2.destroyWindow('image')
            cv2.destroyWindow('cropped')
            select_rectangle = True
            point1, point2 = None, None
            
            return cropped_img

        cv2.imshow("image", img)

# Clean up
cv2.destroyAllWindows()
    

if __name__ == '__main__':
    print('----------------------------------------------------------')
    print('Use left and right mouse buttons to define a rectangle over the object')
    print('Press `s` to crop the selected area')
    print('Press `r` to restart')
    print('Press `d` to return the cropped img')
    print('Press `q` to exit the program')
    print('----------------------------------------------------------')

    main()