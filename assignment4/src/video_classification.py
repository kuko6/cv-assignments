import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
import torch.nn.functional as F

from model import Model
from utils import preprocess_img
from select_rectangle import main as select_rectangle


def capture_video(model: Model, labels):
    cap = cv2.VideoCapture(0) 
    motion_threshold = 1000

    last_frame = None
    motionless_frames = 0

    paused = False

    while True:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # motion detection - frame differencing
        if last_frame is None:
            last_frame = gray
        else:
            diff = cv2.absdiff(last_frame, gray)
            _, motion_mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            motion_pixel_count = cv2.countNonZero(motion_mask)
            # cv2.imshow('motion', motion_mask)
            
            if motion_pixel_count < motion_threshold:
                motionless_frames += 1
            else:
                last_frame = gray
                motionless_frames = 0
            
        # check if motion has stopped for a certain number of frames
        if motionless_frames == 5:
            img = select_rectangle(frame)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = preprocess_img(img)
            # cv2.imshow('freeze', img[0].permute(1,2,0).numpy())

            # perform inference
            probs = model(img.to('mps'))
            probs = F.softmax(probs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            print(f'predicted: {pred}, {labels[str(pred)]} with: {probs[0][pred]:>5f}')
            
            # display
            cv2.putText(frame, f'predicted: {pred}, {labels[str(pred)]} with: {probs[0][pred]:>5f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("frame", frame)

            # pause the video
            if not paused:
                paused = True
                if cv2.waitKey(0): # wait until a key is pressed
                    paused = False  

        # exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    model = Model(double=False)
    checkpoint = torch.load('models/not_doubled_best.pt')
    # checkpoint = torch.load('outputs/best_checkpoint.pt')
    model.load_state_dict(checkpoint['model_state'])
    model.to('mps')
    model.eval()

    with open('data/labels.json', 'r') as fp:
        labels = json.load(fp)

    capture_video(model, labels)


if __name__ == '__main__':
    main()