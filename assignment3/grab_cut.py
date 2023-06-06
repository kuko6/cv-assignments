import cv2
import copy
import numpy as np

# original_img = cv2.imread("data/segmentation/ventricle_segmentation/1.png")
# original_img = cv2.imread("data/segmentation/JPEGImages/2007_000068.jpg") # bird
# original_img = cv2.imread("data/segmentation/JPEGImages/2007_000063.jpg") # dog
original_img = cv2.imread("data/segmentation/JPEGImages/2007_000033.jpg")
# original_img = cv2.imread("data/segmentation/cat.jpg")
# original_img = cv2.imread("data/segmentation/mononoke.jpg")
# original_img = cv2.imread("data/segmentation/butterfly.jpeg")
print(original_img.shape)

if original_img.shape[0] > 1000:
    original_img = cv2.resize(original_img, (original_img.shape[1]//3, original_img.shape[0]//3), interpolation=cv2.INTER_NEAREST)

print(original_img.shape)

img = copy.deepcopy(original_img)
mask = np.zeros(img.shape[:2], np.uint8)
point1 = None
point2 = None

# modes
select_rectangle = True
draw_scribble = False
drawing_left = False
drawing_right = False

# callback function for mouse events
# https://docs.opencv.org/3.4/db/d5b/tutorial_py_mouse_handling.html
def mouse_callback(event, x, y, flags, param):
    global img, point1, point2, rect, select_rectangle, draw_scribble, drawing_left, drawing_right
    
    # left mouse button events
    if event == cv2.EVENT_LBUTTONDOWN:
        if select_rectangle:
            point1 = (x, y)
            img = cv2.circle(img, point1, 4, (0, 0, 255), -1)
        elif draw_scribble:
            drawing_left = True
            print('Drawing foreground')

    elif event == cv2.EVENT_LBUTTONUP:
        drawing_left = False
        
    # right mouse button events
    elif event == cv2.EVENT_RBUTTONDOWN:
        if select_rectangle:
            point2 = (x, y)
            img = cv2.circle(img, point2, 4, (255, 0, 0), -1)
        elif draw_scribble:
            drawing_right = True
            print('Drawing background')
    
    elif event == cv2.EVENT_RBUTTONUP:
        drawing_right = False

    # mouse move events 
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_left and draw_scribble:
            cv2.circle(mask, (x, y), 4, cv2.GC_FGD, -1)
            cv2.circle(img, (x, y), 4, (0, 0, 255), -1)

        elif drawing_right and draw_scribble:
            cv2.circle(mask, (x, y), 4, cv2.GC_BGD, -1)
            cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
        
    # when both points are set, draw a rectangle connecting them
    if point1 and point2 and select_rectangle:
        rect = (min(point1[0], point2[0]), min(point1[1], point2[1]), abs(point1[0] - point2[0]), abs(point1[1] - point2[1]))
        cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)
        select_rectangle = False
        draw_scribble = True


# segmentation with grab cut
# https://docs.opencv.org/3.4/d8/d83/tutorial_py_grabcut.html
def segment(img, mask, rect):
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # use the mask when it contains scribbles
    if len(np.unique(mask)) > 1:
        print('Segmenting using mask')
        mode = cv2.GC_INIT_WITH_MASK
    else:
        print('Segmenting using rectangle')
        mode = cv2.GC_INIT_WITH_RECT
    mask, _, _ = cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, mode)
    
    # mask include values from 0 to 4, (background, foreground, possible background and possible forground)
    # change all background (0, 2) pixels to 0 and all foreground (1, 3) to 1
    tmp_mask = np.where((mask==0) | (mask==2), 0, 255).astype('uint8')
    img = cv2.bitwise_and(img, img, mask=tmp_mask)

    cv2.imshow("seg", img)
    cv2.imshow("mask", tmp_mask)

    return img, mask


def main():
    global img, original_img, mask, point1, point2, rect, select_rectangle, draw_scribble

    print('----------------------------------------------------------')
    print('Use left and right mouse buttons to define a rectangle over the object')
    print('Press `s` to segment specified area')
    print('Hold left and right mouse buttons to specify foreground and background pixels')
    print('Press `r` to restart ')
    print('Press `Esc` to exit the program')
    print('----------------------------------------------------------')

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", mouse_callback)

    while True:
        k = cv2.waitKey(20)

        if k & 0xFF == 27:
            break

        elif k == ord('r'):
            point1, point2, rect = None, None, None
            img = copy.deepcopy(original_img)
            mask = np.zeros(img.shape[:2], np.uint8)
            select_rectangle = True
            draw_scribble = False
        
        elif k == ord('s') and select_rectangle == False:
            seg, mask = segment(original_img, mask, rect)

        elif k == ord('t'):
            draw_scribble = True
            select_rectangle = False

        # if k != -1:
        #     print(k)

        cv2.imshow("image", img)

# Clean up
cv2.destroyAllWindows()
    

if __name__ == '__main__':
    main()