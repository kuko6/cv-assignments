import cv2
import numpy as np
import copy
import math

'''
Based on the opencv tutorial on optical flow: https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html
'''

def sparse_flow(draw_lines=True, save=False):
    ''' Calculates sparse optical flow with `goodFeaturesToTrack()` and `calcOpticalFlowPyrLK()` '''

    vid = cv2.VideoCapture('data/motion/dataset_kaggle/crosswalk.avi')
    ret, frame = vid.read()

    # calculate the first set of keypoints
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp0 = cv2.goodFeaturesToTrack(old_gray, mask=None, qualityLevel=0.01, maxCorners=300, minDistance=10)
    lines_mask = np.zeros_like(frame) 

    dist_threshold = 0.3
    colour = np.random.randint(0, 255, (400, 3))

    if save:
        out = cv2.VideoWriter('outputs/sparse_flow.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame.shape[1],frame.shape[0]))

    while(vid.isOpened()):
        ret, frame = vid.read()
        if not ret or (cv2.waitKey(30) & 0xff == 27): 
            break 
        
        # generate new keypoints 
        if len(kp0) < 5:
            new_kp = cv2.goodFeaturesToTrack(gray, mask=None, qualityLevel=0.05, maxCorners=200, minDistance=30)
            kp0 = np.concatenate((kp0, new_kp), axis=0)

        # calculate optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, kp0, None, winSize=(21, 21), maxLevel=5, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.01))
        if kp1 is not None:
            good_new = kp1[st==1]
            good_old = kp0[st==1]

        # calculate diff between points
        diff = []
        for i, _ in enumerate(good_new):
            diff.append(math.dist(good_new[i], good_old[i]))

        # draw points
        moving_kp = []
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            # throw away stationary keypoints
            if diff[i] > dist_threshold:
                if draw_lines:
                    lines_mask = cv2.line(lines_mask, (int(a), int(b)), (int(c), int(d)), colour[i].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, colour[i].tolist(), -1)
                moving_kp.append(new)
        print('remaining keypoints: ', len(moving_kp))
        
        # draw bounding box 
        if len(moving_kp) > 1:
            x, y, w, h = cv2.boundingRect(np.array(moving_kp))
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        kp0 = np.array(moving_kp).reshape(-1, 1, 2)
        frame = cv2.add(frame, lines_mask)
        cv2.imshow('frame', frame)
        
        if save: 
            out.write(frame)

        old_gray = gray

    vid.release()


def background_sub_segmentation(save=False):
    ''' Background segmentation with the MOG and KNN background substraction methods '''
    
    vid = cv2.VideoCapture('data/motion/dataset_kaggle/crosswalk.avi')
    ret, frame = vid.read()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mog = cv2.createBackgroundSubtractorMOG2(varThreshold=25, detectShadows=True)
    knn = cv2.createBackgroundSubtractorKNN(dist2Threshold=300, detectShadows=True)

    if save:
        out = cv2.VideoWriter('outputs/background_sub_ref.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame.shape[1],frame.shape[0]))
        out_mog = cv2.VideoWriter('outputs/mask_mog.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame.shape[1],frame.shape[0]))
        out_knn = cv2.VideoWriter('outputs/mask_knn.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame.shape[1],frame.shape[0]))

    while(vid.isOpened()):
        ret, frame = vid.read()
        if not ret or (cv2.waitKey(30) & 0xff == 27): 
            break 

        mask_mog = mog.apply(frame)
        mask_mog = cv2.morphologyEx(mask_mog, cv2.MORPH_OPEN, kernel)
        mask_mog = cv2.dilate(mask_mog, kernel)
        # print(mask_mog.shape)
        # print(np.unique(mask_mog))

        mask_knn = knn.apply(frame)
        mask_knn = cv2.morphologyEx(mask_knn, cv2.MORPH_OPEN, kernel)
        mask_knn = cv2.dilate(mask_knn, kernel)
        # print(mask_knn.shape)
        # print(np.unique(mask_knn))

        # print(frame.shape)

        result = cv2.bitwise_and(frame, frame, mask=mask_mog)
        
        cv2.imshow('result', cv2.resize(result, (frame.shape[1]//3, frame.shape[0]//3), interpolation=cv2.INTER_NEAREST))
        cv2.imshow('mask mog', cv2.resize(mask_mog, (frame.shape[1]//3, frame.shape[0]//3), interpolation=cv2.INTER_NEAREST))
        cv2.imshow('mask knn', cv2.resize(mask_knn, (frame.shape[1]//3, frame.shape[0]//3), interpolation=cv2.INTER_NEAREST))
        cv2.imshow('frame', cv2.resize(frame, (frame.shape[1]//3, frame.shape[0]//3), interpolation=cv2.INTER_NEAREST))

        if save: 
            out_mog.write(cv2.cvtColor(mask_mog, cv2.COLOR_GRAY2BGR))
            out_knn.write(cv2.cvtColor(mask_knn, cv2.COLOR_GRAY2BGR))
            out.write(frame)

    vid.release()


def get_bounding_box(mask, method=''):
    ''' 
    Create bounding box for the specified mask 
    @method can be either `''` or `'contours'`
    '''
    
    box = None

    if method == '':
        x, y, w, h = cv2.boundingRect(mask)
        box = {'x': x, 'y': y, 'w': w, 'h': h}
        
    elif method == 'contours':
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            box = {'x': x, 'y': y, 'w': w, 'h': h}
    
    return box


def background_sub_motion_tracking(save=False):
    ''' Motion tracking with the MOG and KNN background substraction methods '''

    vid = cv2.VideoCapture('data/motion/dataset_kaggle/night.avi')
    ret, frame = vid.read()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mog = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    knn = cv2.createBackgroundSubtractorKNN(detectShadows=True)

    if save:
        out = cv2.VideoWriter('outputs/background_sub_tracking_night.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame.shape[1],frame.shape[0]))

    while(vid.isOpened()):
        ret, frame = vid.read()
        if not ret or (cv2.waitKey(30) & 0xff == 27): 
            break 

        # create masks with MOG and KNN
        mask_mog = mog.apply(frame)
        mask_mog = cv2.morphologyEx(mask_mog, cv2.MORPH_OPEN, kernel)
        mask_mog = cv2.dilate(mask_mog, kernel)

        mask_knn = knn.apply(frame)
        mask_knn = cv2.morphologyEx(mask_knn, cv2.MORPH_OPEN, kernel)
        mask_knn = cv2.dilate(mask_knn, kernel)

        # draw bounding boxes
        box_mog = get_bounding_box(mask_mog, method='contours')
        if box_mog != None:
            frame = cv2.rectangle(
                frame, (box_mog['x'], box_mog['y']), (box_mog['x'] + box_mog['w'], box_mog['y'] + box_mog['h']), (0, 255, 0), 2
            )

        box_knn = get_bounding_box(mask_knn, method='contours')
        if box_knn != None:
            frame = cv2.rectangle(
                frame, (box_knn['x'], box_knn['y']), (box_knn['x'] + box_knn['w'], box_knn['y'] + box_knn['h']), (0, 0, 255), 2
            )
        
        cv2.imshow('mask mog', cv2.resize(mask_mog, (frame.shape[1]//3, frame.shape[0]//3), interpolation=cv2.INTER_NEAREST))
        cv2.imshow('mask knn', cv2.resize(mask_knn, (frame.shape[1]//3, frame.shape[0]//3), interpolation=cv2.INTER_NEAREST))
        cv2.imshow('frame', cv2.resize(frame, (frame.shape[1]//3, frame.shape[0]//3), interpolation=cv2.INTER_NEAREST))

        if save: 
            out.write(frame)
    
    vid.release()


def running_average(save=False):
    ''' Background segmentation and motion tracking based on running average '''
    
    vid = cv2.VideoCapture('data/motion/dataset_kaggle/night.avi')
    ret, frame = vid.read()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    running_avg = np.float32(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    if save:
        out = cv2.VideoWriter('outputs/running_average_night.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame.shape[1],frame.shape[0]))
        out_mask = cv2.VideoWriter('outputs/running_average_mask_night.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame.shape[1],frame.shape[0]))

    while(vid.isOpened()):
        ret, frame = vid.read()
        if not ret or (cv2.waitKey(30) & 0xff == 27): 
            break 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # if running_avg is None:
        #     running_avg = np.float32(gray)

        # update the running average
        cv2.accumulateWeighted(gray, running_avg, alpha=0.4)

        # compute the difference between the original frame and the running average
        diff = cv2.absdiff(gray, cv2.convertScaleAbs(running_avg))

        # create binary mask from the computed difference
        _, mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # draw bounding box
        box = get_bounding_box(mask, method='contours')
        if box != None:
            frame = cv2.rectangle(frame, (box['x'], box['y']), (box['x'] + box['w'], box['y'] + box['h']), (0, 255, 0), 2)

        cv2.imshow('blend', cv2.convertScaleAbs(running_avg))
        cv2.imshow('diff', cv2.resize(diff, (frame.shape[1]//3, frame.shape[0]//3), interpolation=cv2.INTER_NEAREST))
        cv2.imshow('mask', cv2.resize(mask, (frame.shape[1]//3, frame.shape[0]//3), interpolation=cv2.INTER_NEAREST))
        cv2.imshow('frame', cv2.resize(frame, (frame.shape[1]//3, frame.shape[0]//3), interpolation=cv2.INTER_NEAREST))

        if save: 
            out_mask.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
            out.write(frame)

    vid.release()


def sparse_flow_background_sub(draw_lines=True, save=False):
    ''' 
    Calculates sparse optical flow with `goodFeaturesToTrack()` and `calcOpticalFlowPyrLK()`
    based on a binary mask created by the MOG background substractor
    '''

    vid = cv2.VideoCapture('data/motion/dataset_kaggle/night.avi')
    ret, frame = vid.read()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mog = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    # create mask for the first frame
    mask_mog = mog.apply(frame)
    mask_mog = cv2.morphologyEx(mask_mog, cv2.MORPH_OPEN, kernel)
    mask_mog = cv2.dilate(mask_mog, kernel)

    # calculate the first set of keypoints
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp0 = cv2.goodFeaturesToTrack(old_gray, mask=mask_mog, qualityLevel=0.01, maxCorners=300, minDistance=10)
    mask = np.zeros_like(frame)

    dist_threshold = 0.3
    colour = np.random.randint(0, 255, (400, 3))

    if save:
        out = cv2.VideoWriter('outputs/sparse_flow_mog_night.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame.shape[1],frame.shape[0]))

    while(vid.isOpened()):
        ret, frame = vid.read()
        if not ret or (cv2.waitKey(30) & 0xff == 27): 
            break 
        
        # create new mask
        mask_mog = mog.apply(frame)
        mask_mog = cv2.morphologyEx(mask_mog, cv2.MORPH_OPEN, kernel)
        mask_mog = cv2.dilate(mask_mog, kernel)

        # generate new keypoints 
        if len(kp0) < 10:
            new_kp = cv2.goodFeaturesToTrack(gray, mask=mask_mog, qualityLevel=0.4, maxCorners=30, minDistance=100)
            kp0 = np.concatenate((kp0, new_kp), axis=0)

        # calculate optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp1, st, _ = cv2.calcOpticalFlowPyrLK(old_gray, gray, kp0, None, winSize=(21, 21), maxLevel=5, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.01))
        if kp1 is not None:
            good_new = kp1[st==1]
            good_old = kp0[st==1]

        # calculate diff between points
        diff = []
        for i, _ in enumerate(good_new):
            diff.append(math.dist(good_new[i], good_old[i]))

        # draw points
        moving_kp = []
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            
            # throw away stationary keypoints
            if diff[i] > dist_threshold:
                if draw_lines:
                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), colour[i].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, colour[i].tolist(), -1)
                moving_kp.append(new)
        print('remaining keypoints: ', len(moving_kp))
        
        # draw bounding box 
        if len(moving_kp) > 1:
            x, y, w, h = cv2.boundingRect(np.array(moving_kp))
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        kp0 = np.array(moving_kp).reshape(-1, 1, 2)
        frame = cv2.add(frame, mask)
        
        cv2.imshow('mask', mask_mog)
        cv2.imshow('frame', frame)

        if save: 
            out.write(frame)

        old_gray = gray.copy()

    vid.release()


def dense_flow(save=False):
    ''' Calculate dense optical flow with `calcOpticalFlowFarneback()` '''

    # vid = cv2.VideoCapture('data/motion/AVG-TownCentre-raw.mp4')
    vid = cv2.VideoCapture('data/motion/PETS09-S2L1-raw.mp4')
    ret, frame = vid.read()
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    threshold = 1

    hsv = np.zeros_like(frame)

    if save:
        out = cv2.VideoWriter('outputs/dense_flow2.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame.shape[1],frame.shape[0]))
        out_mask = cv2.VideoWriter('outputs/dense_flow2_mask.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame.shape[1],frame.shape[0]))
        out_visualization = cv2.VideoWriter('outputs/dense_flow2_visualization.mp4', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame.shape[1],frame.shape[0]))

    while(vid.isOpened()):
        ret, frame = vid.read()
        if not ret or (cv2.waitKey(30) & 0xff == 27): 
            break 

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            old_gray, gray, None, pyr_scale=0.5, levels=4, winsize=17, iterations=3, poly_n=7, poly_sigma=1.7, flags=0
        )

        # get magnitute (speed motion) and angle (direction of motion)
        magnitude, angle = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])

        # create binary mask by thresholding the magnitude
        _, mask = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # create visualization (https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html)
        hsv[:,:,0] = angle * 180 / np.pi / 2 
        hsv[:,:,1] = 255
        hsv[:,:,2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        visualization = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # combine or blend the visualized optical flow and the original frame
        combined = cv2.addWeighted(frame, 1, visualization, 1.5, 0)

        # get suitable contours from the mask
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                filtered_contours.append(contour)
        
        # draw bounding boxes around the contours
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('combined', cv2.resize(combined, (frame.shape[1]//2, frame.shape[0]//2), interpolation=cv2.INTER_NEAREST))
        cv2.imshow('mask', cv2.resize(mask, (frame.shape[1]//2, frame.shape[0]//2), interpolation=cv2.INTER_NEAREST))
        cv2.imshow('flow', cv2.resize(visualization, (frame.shape[1]//2, frame.shape[0]//2), interpolation=cv2.INTER_NEAREST))
        cv2.imshow('frame', cv2.resize(frame, (frame.shape[1]//2, frame.shape[0]//2), interpolation=cv2.INTER_NEAREST))

        if save:
            out.write(frame)
            out_mask.write(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
            out_visualization.write(combined)
        
        old_gray = gray
    
    vid.release()


if __name__ == '__main__':
    save = False

    # sparse_flow(draw_lines=True, save=save)

    # background_sub_segmentation(save=save)
    # running_average(save=save)
    # background_sub_motion_tracking(save=save)
    # sparse_flow_background_sub(draw_lines=True, save=save)

    dense_flow(save=save)

    cv2.waitKey()
    cv2.destroyAllWindows()